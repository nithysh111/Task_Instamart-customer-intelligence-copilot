"""Part B: RFM segmentation and a leakage-aware lapse model."""
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
ROOT=Path(__file__).resolve().parents[1]; CLEAN=ROOT/"data/clean"; OUT=ROOT/"outputs"
SNAPSHOT=pd.Timestamp("2025-10-01")
def load():
    c=pd.read_csv(CLEAN/"customers.csv"); o=pd.read_csv(CLEAN/"orders.csv")
    o["order_ts"]=pd.to_datetime(o.order_ts)
    return c,o
def rfm_and_segments(k=4):
    c,o=load(); done=o[o.status.eq("completed")].copy()
    rfm=done.groupby("customer_id").agg(last_order=("order_ts","max"),frequency=("order_id","nunique"),monetary=("order_amount","sum")).reset_index()
    rfm["recency"]=(SNAPSHOT-rfm.last_order).dt.days; rfm=rfm.drop(columns="last_order")
    scaler=StandardScaler(); X=scaler.fit_transform(rfm[["recency","frequency","monetary"]])
    diagnostics=[]
    from sklearn.metrics import silhouette_score
    for n in range(2,9):
        km=KMeans(n_clusters=n,n_init=10,random_state=42).fit(X)
        diagnostics.append({"k":n,"inertia":km.inertia_,"silhouette":silhouette_score(X,km.labels_)})
    km=KMeans(n_clusters=k,n_init=10,random_state=42); rfm["segment_id"]=km.fit_predict(X)
    profile=rfm.groupby("segment_id").agg(size=("customer_id","size"),recency=("recency","mean"),frequency=("frequency","mean"),monetary=("monetary","mean")).reset_index()
    order=profile.sort_values("monetary",ascending=False).segment_id.tolist()
    names=["Premium" if i==order[0] else ("Frequent" if profile.set_index("segment_id").loc[i,"frequency"]>=profile.frequency.median() else "At Risk") for i in profile.segment_id]
    name_map=dict(zip(profile.segment_id,names))
    rfm["segment_name"]=rfm.segment_id.map(name_map); rfm["is_premium"]=rfm.segment_id.eq(order[0])
    OUT.mkdir(exist_ok=True)
    rfm[["customer_id","recency","frequency","monetary","segment_id","segment_name","is_premium"]].to_csv(OUT/"customer_segments.csv",index=False)
    pca=PCA(n_components=2).fit(X)
    return rfm,pd.DataFrame(diagnostics),profile,pca.explained_variance_ratio_
def build_lapse_model():
    c,o=load(); cutoff=pd.Timestamp("2025-09-01"); end=pd.Timestamp("2025-10-01")
    hist=o[(o.order_ts<cutoff)].copy(); done=hist[hist.status.eq("completed")]
    cohort=done.groupby("customer_id").agg(last_order=("order_ts","max"),frequency=("order_id","nunique"),monetary=("order_amount","sum"),
      avg_discount=("discount_pct","mean"),avg_delivery=("delivery_minutes","mean")).reset_index()
    cohort["recency"]=(cutoff-cohort.last_order).dt.days; cohort=cohort.drop(columns="last_order")
    all_final=hist[hist.status.isin(["completed","cancelled"])].groupby("customer_id").status.agg(total="count",cancelled=lambda x:(x=="cancelled").sum()).reset_index()
    cohort=cohort.merge(all_final,on="customer_id",how="left"); cohort["cancellation_rate"]=cohort.cancelled/cohort.total
    cats=done.merge(pd.read_csv(CLEAN/"products.csv")[["product_id","category"]],on="product_id").groupby(["customer_id","category"]).size().rename("n").reset_index()
    fav=cats.sort_values(["customer_id","n","category"],ascending=[True,False,True]).drop_duplicates("customer_id").rename(columns={"category":"favourite_category"})[["customer_id","favourite_category"]]
    cohort=cohort.merge(fav,on="customer_id",how="left").merge(c,on="customer_id",how="left")
    cohort["signup_date"]=pd.to_datetime(cohort.signup_date); cohort["tenure_days"]=(cutoff-cohort.signup_date).dt.days
    post=o[(o.order_ts>=cutoff)&(o.order_ts<end)&o.status.eq("completed")].groupby("customer_id").size()
    cohort["lapsed"]=(~cohort.customer_id.isin(post.index)).astype(int)
    cohort=cohort.drop(columns=["signup_date","total","cancelled"])
    features=["recency","frequency","monetary","avg_discount","cancellation_rate","avg_delivery","favourite_category","city","age","gender","tenure_days"]
    X=cohort[features].copy(); y=cohort.lapsed
    customer_ids=cohort.customer_id.copy()
    numeric=["recency","frequency","monetary","avg_discount","cancellation_rate","avg_delivery","age","tenure_days"]
    categorical=["favourite_category","city","gender"]
    X_train,X_test,y_train,y_test,ids_train,ids_test=train_test_split(
        X,y,customer_ids,test_size=.2,stratify=y,random_state=42)
    class_distribution={int(label):int(count) for label,count in y.value_counts().sort_index().items()}
    training_distribution={int(label):int(count) for label,count in y_train.value_counts().sort_index().items()}
    testing_distribution={int(label):int(count) for label,count in y_test.value_counts().sort_index().items()}
    prep=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),numeric),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="constant",fill_value="unknown")),
                         ("encoder",OneHotEncoder(drop="first",handle_unknown="ignore"))]),categorical)])
    models={}
    estimators=[
        ("logistic",LogisticRegression(max_iter=1000),{"model__C":[.01,.1,1,10,100]}),
        ("forest",RandomForestClassifier(n_estimators=100,random_state=42),{"model__max_depth":[3,5,8],"model__min_samples_leaf":[5,10,20]})]
    for name,est,grid in estimators:
        pipe=Pipeline([("prep",prep),("model",est)])
        search=GridSearchCV(pipe,grid,cv=5,scoring="f1",n_jobs=-1); search.fit(X_train,y_train)
        models[name]={"search":search}
    forest_grid=estimators[1][2]
    sampled_pipe=ImbalancedPipeline([
        ("prep",prep),
        ("sampler",RandomOverSampler(random_state=42)),
        ("model",RandomForestClassifier(n_estimators=100,random_state=42))])
    sampled_search=GridSearchCV(sampled_pipe,forest_grid,cv=5,scoring="f1",n_jobs=-1)
    sampled_search.fit(X_train,y_train)
    models["forest_oversampled"]={"search":sampled_search}

    metrics={}
    for name,entry in models.items():
        search=entry["search"]
        pred=search.predict(X_test)
        metrics[name]={"best_params":search.best_params_,"cv_f1":float(search.best_score_),
          "accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred,zero_division=0),
          "recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),
          "confusion_matrix":confusion_matrix(y_test,pred).tolist()}
    selected_name=max(["forest","forest_oversampled"],key=lambda name:metrics[name]["cv_f1"])
    selected=models[selected_name]["search"]
    OUT.mkdir(exist_ok=True); joblib.dump(selected,OUT/"lapse_model.joblib")
    reloaded=joblib.load(OUT/"lapse_model.joblib")
    assert np.array_equal(selected.predict(X_test),reloaded.predict(X_test))
    historical=done[done.order_ts<cutoff].groupby("customer_id").agg(
        frequency=("order_id","nunique"),monetary=("order_amount","sum"),
        last_order=("order_ts","max")).reset_index()
    historical["recency"]=(cutoff-historical.last_order).dt.days
    historical_features=["recency","frequency","monetary"]
    historical_scaled=StandardScaler().fit_transform(historical[historical_features])
    historical_clusters=KMeans(n_clusters=4,n_init=10,random_state=42).fit_predict(historical_scaled)
    historical["segment_id"]=historical_clusters
    premium_cluster=historical.groupby("segment_id").monetary.mean().idxmax()
    premium=set(historical.loc[historical.segment_id.eq(premium_cluster),"customer_id"])
    probs=selected.predict_proba(X_test)[:,list(selected.classes_).index(1)]
    ret=pd.DataFrame({"customer_id":ids_test.to_numpy(),
        "lapse_probability":probs,"actually_lapsed":y_test.values})
    ret=ret[ret.customer_id.isin(premium)].sort_values("lapse_probability",ascending=False).head(10)
    ret.to_csv(OUT/"retention_list.csv",index=False)
    fitted=selected.best_estimator_.named_steps["model"]
    feature_names=selected.best_estimator_.named_steps["prep"].get_feature_names_out()
    if hasattr(fitted,"feature_importances_"):
        importance_values=fitted.feature_importances_
        interpretation="Random-forest impurity-based feature importance; larger values indicate greater contribution to the fitted trees."
    else:
        importance_values=np.abs(fitted.coef_).mean(axis=0)
        interpretation="Mean absolute logistic-regression coefficient; larger values indicate stronger association, not causation."
    importance=pd.DataFrame({"feature":feature_names,"importance":importance_values})
    importance=importance.sort_values("importance",ascending=False).reset_index(drop=True)
    return {"cohort_rows":len(cohort),"lapsed":int(y.sum()),
        "baseline_accuracy":accuracy_score(y_test,np.zeros(len(y_test),dtype=int)),
        "class_distribution":class_distribution,"training_distribution":training_distribution,
        "testing_distribution":testing_distribution,
        "selected_model":selected_name,"models":metrics,"feature_importance":importance,
        "feature_importance_explanation":interpretation,"retention_list":ret,
        "retention_eligibility_as_of":cutoff.strftime("%Y-%m-%d")}
