# imported the requests library 
import requests 


irenebuoy = ["42060","41043","41046","mlrf1","fwyf1","spgf1","41004","41013","41025","44014","44009","44025","44020"] #bouynumber array
year = "2011" #year

bouynumber = irenebuoy

for i in bouynumber :
	text_url = "https://www.ndbc.noaa.gov/view_text_file.php?filename="+i+"h"+year+".txt.gz&dir=data/historical/stdmet/"
	  

	r = requests.get(text_url) # create HTTP response object 
	  
	# send a HTTP request to the server and save 
	# the HTTP response in a response object called r 

	with open(i+".txt",'wb') as f: 
	  
	    # Saving received content as a text file in 
	    # binary format 
	    # write the contents of the response (r.content) 
	    # to a new file in binary mode. 
	    f.write(r.content) 





import pandas as pd 
maindf = pd.DataFrame()

windspeedfinal = []
pressurefinal = []
for i in bouynumber :

	f=open(i+'.txt','r')

	
	windspeed = []
	pressure = []
	for line in f :
		windspeed.append(line[21:25])
		pressure.append(line[53:59])

	windspeedfinal = windspeedfinal  + windspeed[2:]	
	pressurefinal = pressurefinal  + pressure[2:]


maindf['windspeed'] = windspeedfinal ##from 2 to the end so as to ignore the title lables
maindf['pressure'] = pressurefinal

maindf.to_csv("IRENECSV.csv")
