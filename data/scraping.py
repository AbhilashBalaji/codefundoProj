# imported the requests library 
import requests 
import pandas as pd 

irenebuoy = ["42060","41043","41046","mlrf1","fwyf1","spgf1","41004","41013","41025","44014","44009","44025","44020"] #bouynumber array
year = "2011" #year


testdf = pd.read_csv("stations.csv")
stationdic = {}


for index,row in testdf.iterrows() :
	ID  = testdf.loc[index,'ID']
	if ID in irenebuoy :
		if ID not in stationdic.keys():
			stationdic[ID] = (testdf.loc[index,'Lat'],testdf.loc[index,'Lon'])



print(stationdic)

for i in irenebuoy :
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
yearfinal  =[]
monthfinal =[]
dayfinal =[]
hourfinal = []
minsfinal = []
waveheightfinal = []
waveperiodfinal = []
wavedirectionfinal = []
latfinal = []
lonfinal = []
idlistfinal = []
for i in irenebuoy :

	f=open(i+'.txt','r')

	
	windspeed = []
	pressure = []
	year  =[]
	month =[]
	day =[]
	hour = []
	mins = []
	waveheight = []
	waveperiod = []
	wavedirection = []
	lat = []
	lon = []
	idlist = []
	for line in f :
		windspeed.append(line[21:25])
		pressure.append(line[53:59])
		year.append(line[0:5])
		month.append(line[5:8])
		day.append(line[8:11])
		hour.append(line[11:14])
		mins.append(line[14:17])
		waveheight.append(line[32:36])
		waveperiod.append(line[44:48])
		wavedirection.append(line[49:52])
		lat.append(stationdic[i][0])
		lon.append(stationdic[i][1])
		idlist.append(i)
	windspeedfinal = windspeedfinal  + windspeed[2:]	
	pressurefinal = pressurefinal  + pressure[2:]
	yearfinal = yearfinal  + year[2:]	
	monthfinal = monthfinal  + month[2:]
	dayfinal = dayfinal  + day[2:]	
	hourfinal = hourfinal  + hour[2:]
	minsfinal = minsfinal  + mins[2:]	
	waveheightfinal = waveheightfinal  + waveheight[2:]	
	waveperiodfinal = waveperiodfinal + waveperiod[2:]
	wavedirectionfinal = wavedirectionfinal + wavedirection[2:]
	latfinal  = latfinal  + lat[2:]
	lonfinal = lonfinal  + lon[2:]
	idlistfinal = idlistfinal + idlist[2:]
	#pressurefinal = pressurefinal  + pressure[2:]


maindf['ID'] = idlistfinal
maindf['year'] = yearfinal
maindf['month'] = monthfinal
maindf['day'] = dayfinal
maindf['hour'] = hourfinal
maindf['mins'] = minsfinal
maindf['waveperiod'] = waveperiodfinal
maindf['wavedirection'] = wavedirectionfinal
maindf['waveheight'] = waveheightfinal
maindf['Lat'] = latfinal
maindf['Lon'] = lonfinal


maindf['windspeed'] = windspeedfinal ##from 2 to the end so as to ignore the title lables
maindf['pressure'] = pressurefinal



maindf.to_csv("IRENECSV.csv")
#print(latfinal)

