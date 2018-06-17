import spark_functions as sf
import cleaning_functions as cf
import exploration_functions as ef

es, sql  = sf.sparkContext()
index = "curated_sensors"
doc_type = "IoT" 
#input
df_spark = sf.elastic2df(es,index,doc_type,sql)
df = df_spark.toPandas()

df = df.set_index('timestamp', drop=True)

ef.heatmap(df)
ef.HowManyBlanks(df)

path = '/home/marcroig/Desktop/data/reports/spark_report.html'
sf.sparkReport(df,path)
path = '/home/marcroig/Desktop/data/reports/pandas_report.html'
ef.pandasReport(df,path)