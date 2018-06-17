#MAIN_DATA_ACQUISITION
import acquition_functions as af

import time
import datetime
from elasticsearch import Elasticsearch
#----------------------------GET JSON SAMPLE---------------------------#

start_time = time.time()
d = datetime.date(2017,11,10)
ending = datetime.date(2017,11,11)
increment = datetime.timedelta(days=1)

variables = 'temperature'

bikes = range(2,3)

for bike in bikes:
	bike_number = bike
	doc_name = 'sample_'+str(bike)+'_'+str(d)+'_'+variables+'.json'
	af.get_json(doc_name,bike_number,d,ending,variables,increment)

print("--- %s seconds ---" % (time.time() - start_time))

#----------------------------GET JSON FULL-----------------------------#

start_time = time.time()
#d = datetime.date(2017,2,7)
#ending = datetime.date(2018,3,2)
d = datetime.date(2017,11,10)
ending = datetime.date(2017,11,11)
increment = datetime.timedelta(days=1)

variables = ['Vsense','co','humidity','light','temperature','Isense',
			'pm','o3','no2','h2','nh3','ch4','c3h8','c4h10','so2',
			'sound','uv','editemp']
bikes = range(1,4)

for bike in bikes:
	bike_number = bike
	doc_name = 'test'+str(bike)+'.json'
	af.get_json(doc_name,bike_number,d,ending,variables,increment)

print("--- %s seconds ---" % (time.time() - start_time))

#-------------------------INGESTION ON ELASTIC-------------------------#

start_time = time.time()
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
index_name = "semi_raw_sensors"

bike1 = 'bike1.json'
bike2 = 'bike2.json'
bike3 = 'bike3.json'

af.semi_raw_ingest_elastic(bike1,1,es,index_name)
af.semi_raw_ingest_elastic(bike2,2,es,index_name)
af.semi_raw_ingest_elastic(bike3,3,es,index_name)

print("--- %s seconds ---" % (time.time() - start_time))
