import spark_df_profiling
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import FloatType
from pyspark.sql import functions as F


def elastic2df(es,index,doc_type,sql):
    df = sql.read.format("org.elasticsearch.spark.sql").load(index+"/"+doc_type)
    return df

def df2elastic(es,index, doc_type,df):
    df.write.format('org.elasticsearch.spark.sql').save(index+"/"+doc_type)

def transform (df):
    #split lat lon
    split_col = F.split(df['location'], ',')
    df=df.withColumn('lat',split_col.getItem(0))
    df=df.withColumn('lon',split_col.getItem(1))
    df=df.drop('location')
    #change type
    df=df.withColumn('lat',df['lat'].cast(FloatType()).alias('lat'))
    df=df.withColumn('lon',df['lon'].cast(FloatType()).alias('lon'))
    #filtered fields
    df = df.selectExpr('sampling_time as timestamp','lat','lon','variable','result')
    return df

def pivoting(df):
    df_pivot = df.groupby(df.timestamp).pivot('variable').min('result')
    df_pivot = df_pivot.drop('uv')
    df_pivot = df_pivot.drop('light')
    counter = df.groupBy(['timestamp','lat','lon']).count().alias('lat_count')
    most_frequent_latlon = counter.groupBy('timestamp').agg(F.max(F.struct(F.col('count'),F.col('lat'),F.col('lon'))).alias('max')).select(F.col('timestamp'),F.col('max.lat'),F.col('max.lon'))
    df_full = df_pivot.join(most_frequent_latlon, 'timestamp', 'full').orderBy(df_pivot.timestamp)
    return df_full


def createReport(df,path):
	#path: /home/marcroig/Desktop/data/reports/filename.html
	report = spark_df_profiling.ProfileReport(df)
	report.to_file(path)

from sklearn.linear_model import LinearRegression,LogisticRegression
import pandas as pd
import numpy as np


	
	





	
#---------------------------------DEPRECATED----------------------------------

    #list of latitudes per timestamp
    #df_lat = df.groupby(df.timestamp).agg(collect_list('lat')).selectExpr('timestamp','collect_list(lat) as lat_list')
    #df_lon = df.groupby(df.timestamp).agg(collect_list('lon')).selectExpr('timestamp','collect_list(lat) as lat_list')

def separateByVariable(df,str_variable):
    df_variable=df.filter(df.variable==str_variable)
    df_variable=df_variable.selectExpr('timestamp as timestamp','lat','lon','result as '+str_variable)
    #duplicates
    df_variable=df_variable.dropDuplicates(['timestamp'])
    return df_variable

def old_pivoting(df):
    df_variables=df.groupBy(df.variable).count()
    list_variables=[i.variable for i in df_variables.select('variable').collect()]
    list_variables.remove('uv')
    list_varialbles.remove('light')
    i=0
    for str_variable in list_variables:
		
        df_variable = separateByVariable(df,str_variable)
        if i==0:
            df_full=df_variable.columns[0,]          
        else:
            df_full=df_full.join(df_variable.timestamp,df_full.timestamp,'fullouter')          
        i=1
    return df_full

