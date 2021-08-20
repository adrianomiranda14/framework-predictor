import pandas as pd



def lower_case(dataframe):
  """ Transforms all values in columns to lower case """
  for col in dataframe.columns:
    if dataframe[col].dtypes == 'object':
      dataframe[col] = dataframe[col].str.lower()
  return dataframe



def clean_columns(path):
  "This cleans column names of the csv passed through"
  df = pd.read_csv(path,  low_memory=False)
  df.columns = [x.lower() for x in df.columns]
  df.columns = [x.replace(" ","_") for x in df.columns]
  df.columns = [x.replace(",","") for x in df.columns]
  df.columns = [x.replace(")","") for x in df.columns]
  df.columns = [x.replace("(","") for x in df.columns]
  #df['job_role'] = df['job_role'].str.lower()
  return df

def job_title_fun(model, string):
  x = vectorizer.transform([string])
  y = model.predict(x)
  return y

def predict_job_title_prob(string):
  x = vectorizer.transform([string])
  y = nb.predict_proba(x)
  z = nb.classes_[np.argsort(-y)]
  a = z[0][0:3]
  return a