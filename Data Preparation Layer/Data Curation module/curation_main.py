#MAIN_DATA_CURATION--------run it with spark-submit 
import time
from elasticsearch import Elasticsearch
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import col
from spark_functions import elastic2df, df2elastic
start_time = time.time()

#spark connexion
sc = SparkSession.builder.appName("sensors processing").getOrCreate()
sqlContext = SQLContext(sc)
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

#---------------------ELASTIC 2 SPARK-----------------------------#

#elasticsearch variables
index = "clean_sensors"
index_output = "curated_sensors"
doc_type = "IoT" 

#spark processing: delete strange float values on results (86 records)
df = elastic2df(es,index,doc_type)
df_float = df.select(col('bike'),
					col('location'),
					df.result.cast('float').alias('result'),
					col('sampling_time'), 
					col('variable'))
df_curation = df_float.where('result is not null')

df2elastic(es,index_output,doc_type,df_curation)

print("--- %s seconds ---" % (time.time() - start_time))

#---------------------------PIVOTING-----------------------------------#
#elasticsearch variables
index = "bike2_sensors"
index_output = "pivoted_sensors"
doc_type = "IoT" 

#input
df = sf.elastic2df(es,index,doc_type,sqlContext)

#transformations
df = sf.transform(df)
df_full = sf.pivoting(df)

sf.df2elastic(es,index_output,doc_type,df_full)

path = '/home/marcroig/Desktop/data/reports/pivoted_report.html'
sf.createReport(df_full,path)


