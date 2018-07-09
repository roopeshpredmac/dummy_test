#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 14:53:33 2018

@author: predmac
"""

import pandas as pd
import config
from datetime import datetime, timedelta
import logging
from joblib import Parallel, delayed

from pyspark.sql import SQLContext

import pymysql
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error as mse
import math

# flag to confirm the writting of forecasted value to db
real_flag = 0
total_t1 = datetime.now()

import os
import sys


from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.sql import SparkSession

def connect_to_mysql():
    connection = pymysql.connect(host = config.db_host,
                            port= config.db_port,
                            user= config.db_user,
                            password= config.db_pass,
                            db= config.db_name,
                            charset='utf8',
                            cursorclass=pymysql.cursors.DictCursor)
    return connection

conn=connect_to_mysql()

full_t1 = datetime.now()
# initialise sparkContext
spark1 = SparkSession.builder \
    .master('local') \
    .appName('p7_sample') \
    .config('spark.executor.memory', '24gb') \
    .config("spark.cores.max", "16") \
    .getOrCreate()

sc = spark1.sparkContext

# using SQLContext to read parquet file

sqlContext = SQLContext(sc)

## Logging ##

newpath = r'log_for_demo' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
newpath = r'log_for_demo/p5' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

if(real_flag==1):    
    newpath = r'p5' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

#for handler in logging.root.handlers[:]:
#    logging.root.removeHandler(handler)
    
logging.basicConfig(filename='log_for_demo/p5/p5.log',level=logging.DEBUG)

from fbprophet import Prophet
from sklearn.metrics import mean_squared_error as mse
import math
from tqdm import tqdm
#z1.index = z1.date
def create_prophet_m(app_name,z1,delay=24):
    
    ### --- For realtime pred ---###
    
    full_df = z1.bw.iloc[0:len(z1)]
    full_df = full_df.reset_index()
    full_df.columns = ['ds','y']
    
    #removing outliers
    q50 = full_df.y.median()
    q100 = full_df.y.quantile(1)
    q75  = full_df.y.quantile(.75)
    
    if((q100-q50) >= (2*q75)):
        
        full_df.loc[full_df.y>=(2*q75),'y'] = None
    
    #-- Realtime prediction --##
    #model 
    model_r = Prophet(yearly_seasonality=False,changepoint_prior_scale=.2)
    model_r.fit(full_df)
    future_r = model_r.make_future_dataframe(periods=delay,freq='H')
    forecast_r = model_r.predict(future_r)
    forecast_r.index = forecast_r['ds']
    #forecast 
    pred_r = pd.DataFrame(forecast_r['yhat'][len(z1):(len(z1)+delay)])
    pred_r=pred_r.reset_index()
    #--- completes realtime pred ---#
    
    train_end_index=len(z1.bw)-delay
    train_df=z1.bw.iloc[0:train_end_index]
    #train_df= train_df[train_df<cutter]
    
    
    test_df=z1.bw.iloc[train_end_index:len(z1)]
    
    
    
    train_df=train_df.reset_index()
    test_df=test_df.reset_index()
    train_df.columns=['ds','y']
    
    #--- removing outliers in trainset  ---#
    
    q50 = train_df.y.median()
    q100 = train_df.y.quantile(1)
    q75  = train_df.y.quantile(.75)
    
    if((q100-q50) >= (2*q75)):
        
        train_df.loc[train_df.y>=(2*q75),'y'] = None
    
    test_df.columns=['ds','y']
    
    #model 
    model = Prophet(yearly_seasonality=False,changepoint_prior_scale=.2)
    model.fit(train_df)
    future = model.make_future_dataframe(periods=len(test_df),freq='H')
    forecast = model.predict(future)
    forecast.index = forecast['ds']
    #forecast 
    pred = pd.DataFrame(forecast['yhat'][train_end_index:len(z1)])
    pred=pred.reset_index()
    pred_df=pd.merge(test_df,pred,on='ds',how='left')
    pred_df.dropna(inplace=True)
    
    df=pd.DataFrame()
    
    if(len(pred_df)>0):
        
        pred_df['error_test']=pred_df.y-pred_df.yhat
    
        
    
        MSE=mse(pred_df.y,pred_df.yhat)
        RMSE=math.sqrt(MSE)
        pred_df['APE']=abs(pred_df.error_test*100/pred_df.y)
        MAPE=pred_df.APE.mean()
        print("App name:",app_name)
        #print("MSE  :",MSE)
        print("RMSE :",RMSE)
        print("MAPE :",MAPE)
        
       
        mape_q98=pred_df['APE'][pred_df.APE<pred_df['APE'].quantile(0.98)].mean()

        df = pd.DataFrame({'length':len(z1),
                             'test_rmse':RMSE,
                             'test_mape':MAPE,
                
                 'test_mape_98':mape_q98},
                   
                          index=[app_name])

    return(df,model,forecast,pred_df,pred_r)

#-- Function to select a combination for the run

def forapp(t,s,a,df):
    
    
    df2 = df[(df.target_address == t)]

    
    df2 = df2[['bw','application','source','target_address','time_stamp']]
    
    df2 = df2.sort_values(by='time_stamp')
    
    
    prophet_df = pd.DataFrame()
    prophet_analysis_df = pd.DataFrame()
    prophet_future_df = pd.DataFrame()
   
   
    if(len(df2)>1400):
            
        t2 = datetime.now()
            
        prophet_analysis_df,ew_model,ew_forcast,prophet_df,prophet_future_df =(create_prophet_m(a,df2,24))
        t2 = datetime.now()
        
        prophet_analysis_df['application'] = a
        prophet_analysis_df['target'] = t
        prophet_analysis_df['source'] = s
            
        prophet_future_df['application'] = a
        prophet_future_df['target'] = t
        prophet_future_df['source'] = s
        
        prophet_df['target'] = t
        prophet_df['source'] = s
        prophet_df['application'] = a
         
   
    
    return prophet_df, prophet_analysis_df, prophet_future_df,df2

# Reading data from parquet
print('satrt quering')

qt1 = datetime.now()
df = sqlContext.read.parquet('appid_datapoint_parquet1')
df = df[df.app_rsp_time!=0]
df = df[df.byte_count!=0]

# Needed data extraction
from sqlalchemy import create_engine

engine = create_engine(str("mysql+pymysql://"+config.db_user+":"+config.db_pass+"@"+config.db_host+":"+str(config.db_port)+"/"+config.db_name))

t1 = datetime.now()

# Checking whether the reference data is available in db or not , if no then creating it
with conn.cursor() as cursor:
        # Read a  record
        sql = "SHOW TABLES LIKE 'reference_df'" 
        cursor.execute(sql)
        result = (cursor.fetchall())

if result:
    with conn.cursor() as cursor:
        # Read a  record
        sql = "select * from reference_df" 
        cursor.execute(sql)
        rdf = pd.DataFrame(cursor.fetchall())
    so_list = list(rdf.source)
    ap_list = list(rdf.application)
    ta_list = list(rdf.target_address)
    #data = df[(df.source == s ) & (df.application==a)]
    
else:
    ap_list = ['DNS','Extreme Networks','CIFS','Skype for Business','DCERPC',
 'Kerberos','LDAP','Outlook Office365','Outlook','Facebook']

    for idx,a in enumerate(ap_list):
        if idx == 0:
            df1 = df[df.application==a]
            dff = df1.limit(50)
        else:
            df1 = df[df.application==a]
            dff=dff.unionAll(df1.limit(50))
        
    ta = dff.select('target_address').collect()
    ta_list = [str(i.target_address) for i in ta]

    ap = dff.select('application').collect()
    ap_list = [str(i.application) for i in ap]

    so = dff.select('source').collect()
    so_list = [str(i.source) for i in so]

    rdf = pd.DataFrame({'application':ap_list,'source':so_list,'target_address':ta_list})
    rdf.drop_duplicates(inplace=True)
    
    
    rdf.to_sql(con=engine, name='reference_df', if_exists='replace')

    
rdf1 = rdf[['application','source']]
rdf1.drop_duplicates(inplace=True)
ap_list = list(rdf1.application)
so_list = list(rdf1.source)

prophet_df = pd.DataFrame()
prophet_future_df = pd.DataFrame()
prophet_analysis_df = pd.DataFrame()
bw_full_df = pd.DataFrame() 

t2 = datetime.now()
time_to_fetch = str(t2-t1)
    
for k in tqdm(range(0,len(ap_list))):
    a = ap_list[k]
    s = so_list[k]
    ta1_list = list(rdf[(rdf.source==s) & (rdf.application==a)].target_address.unique())

    data = df[(df.source == s ) & (df.application==a)]
   
    data = data[data.target_address.isin(ta1_list)]

    # data cleaning
    bw_df=data.toPandas()
    
    
    
    bw_df['bw'] = bw_df['byte_count']/(8*3600)


    bw_df = bw_df.sort_values(by='bw',ascending=True)       
    dates_outlook = pd.to_datetime(pd.Series(bw_df.time_stamp),unit='ms')
    bw_df.index = dates_outlook   
    bw_df = bw_df.sort_values(by='time_stamp')
    #bw_df.to_csv('p5/bw_per_source_per_app_per_target_dataset.csv',index=False)
    print('quering is successfull')



    logging.info(datetime.now())
    logging.info('-I- Fetching query successfull...')

    if(k==0):
        qt2 = datetime.now()
        query_time = str(qt2-qt1)

 
    # Running for all combiantions

    qt1 = datetime.now()


    #prophet_df = pd.DataFrame()
    #prophet_future_df = pd.DataFrame()
    #prophet_analysis_df = pd.DataFrame()


    pool = Parallel(n_jobs=-1,verbose=5,pre_dispatch='all')
    r0  = pool(delayed(forapp)(t,s,a,bw_df) for t in ta1_list) 


    for i in range(0,len(r0)):
        prophet_df = prophet_df.append(r0[i][0])
        prophet_analysis_df = prophet_analysis_df.append(r0[i][1])
        prophet_future_df = prophet_future_df.append(r0[i][2])
        bw_full_df = bw_full_df.append(r0[i][3])
        
    qt2 = datetime.now()
    model_time  = str(qt2-qt1)
  
 
print(' -I- dataframe cteated ')
logging.info(datetime.now())
logging.info('-I- Model ran succesdfully...')

# saving as csv for graphical representation
if(real_flag==1):
    bw_full_df.to_csv('p5/bw_per_source_per_app_per_target_dataset.csv',index=False)
    prophet_analysis_df.to_csv('p5/bw_analysis_per_source_per_app_per_target_data.csv',index=False)
    prophet_df.to_csv('p5/bw_evaluation_per_source_per_app_per_target_data.csv',index=False)
    prophet_future_df.to_csv('p5/bw_forecast_per_source_per_app_per_target_data.csv',index=False)





# Writing the forecasted data to to mysql_db

if(real_flag==1):    
    prophet_future_df.to_sql(con=engine, name='forecast_bw_per_source_per_app_per_target', if_exists='replace')


total_t2 = datetime.now()
# calculating runtime in minuts
total_real = (total_t2 - total_t1).seconds/60
total_time = str(total_t2 - total_t1)

#for analysis of our model in future

prophet_analysis_df['run_date'] = datetime.now().date()
prophet_analysis_df['total_run_time'] = total_real
prophet_analysis_df.index = list(range(0,len(prophet_analysis_df)))

prophet_analysis_df.to_sql(con=engine, name='analyse_p5', if_exists='append')

print(total_time)
## Logging
logging.info(datetime.now())
logging.info('-I- validation of model...')
logging.info(prophet_analysis_df)

logging.info('-I- Run time for fetching the data from parquet file is')
logging.info(query_time)
logging.info('-I- Run time for modelling is ')
logging.info(model_time)
logging.info('-I- The total run time  is ')
logging.info(total_time)
print ('Total run time  is ', total_time)