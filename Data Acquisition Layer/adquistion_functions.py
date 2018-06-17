import json
import datetime
from urllib2 import Request,urlopen
from elasticsearch import Elasticsearch

#---------------------GET RAW DATA ON JSON-----------------------------#
# returns de request in url format
def get_request(bike_number,start_date,end_date,variable):	
	auth = '4db70b047ca9f8c8771778a462912def'
	url1 = 'http://82.223.79.251:8280/t/growsmarter.cellnextelecom/v.2.0'
	url2 = '/growsmarter/observations?sensor=SBICYCLE'+bike_number+'_'+ \
			variable+'&start_date='+start_date+'&end_date='+ \
			end_date+'&fields=location%2Csampling_time%2Cresult'
	url = url1 + url2
	req = Request(url, headers={"Authorization": "Bearer %s" %auth})
	return req

'''returns the following structure: dicc['variable'][0:len(results on 
this day)]['atribute'] where the atributes are: 
results, sampling_time, location['type'] and location['coordinates'][0:3]'''
def get_sensor_data(bike_number, start_date, end_date,variables):
	dicc=dict()
	for variable in variables:	
		req = get_request(bike_number,start_date,end_date,variable)
		response = urlopen(req)
		html = response.read()
		data_variable = json.loads(html)
		dicc[variable] = data_variable
	return dicc
	
#returns an array with an unique bike data for all days
def get_bike(i, d, ending, variables, increment):
	dicc_day = {}
	bike_data = []	
	while d!= ending:
		start_date = d.strftime('%Y/%m/%d')
		d += increment
		end_date = d.strftime('%Y/%m/%d')
		dicc_day= get_sensor_data(str(i),start_date,end_date,variables)
		bike_data.append(dicc_day)
	return bike_data
	
#creates 3 json with all 3 bikes data
def get_json(doc_name,bike,d,ending,variables,increment):
	with open(doc_name,'w') as outfile:
		json.dump(get_bike(bike,d,ending,variables,increment), outfile)

'''--------------------------INGESTION-------------------------------'''

#STRING PRE-MAPPING (deprecated)
def raw_ingest_elastic(doc,bike,elastic,index_name):
	
	data=json.load(open(doc))
	for day in data:
		for variable in day:
			for result in day[variable]:	
				result['variable']=variable
				result['bike']=bike
				res = elastic.index(index=index_name,doc_type="IoT",
					body=result)	
				result = None
			
#STRING PRE-MAPPING				
def semi_raw_ingest_elastic(doc,bike,elastic,index_name):
	data=json.load(open(doc))
	for day in data:
		for variable in day:
			for result in day[variable]:
				result['variable'] = variable
				result['bike'] = bike
				lat = str(result['location']['coordinates'][0])
				lon = str(result['location']['coordinates'][1])
				result['location']=	lat+","+lon
				res = elastic.index(index=index_name,doc_type="IoT",body=result)
				result = None

#NEEDS PRE-MAPPING! (deprecated) delete location.type key, invert order on location.coordinates, delete incorrect results 
def python_curated_ingest_elastic(doc,bike,elastic,index_name):
	nulls=0
	errors=0
	data=json.load(open(doc))
	for day in data:
		for variable in day:
			for result in day[variable]:	
				if result!={}:
					result['variable']=variable
					result['bike']=bike
					try:
						float(result['result'])
						result['location']['coordinates']= result['location']['coordinates'][::-1]
						temp=result['location'].pop('type',None)
						res = elastic.index(index=index_name,doc_type="IoT",body=result)
					except:
						errors +=1
				else:
					nulls +=1
				result = None
	print "QUANTITAT D'ERRORS EN FORMAT" + str(errors)
	print "QUANTITAT DE VALORS NULS:\t" + str(nulls)
				
