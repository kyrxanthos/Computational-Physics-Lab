"""
Ph21 Assignment 1 part 2
Created By Kyriacos Xanthos

"""
import urllib.request
import urllib.parse
import matplotlib.pyplot as plt
from astropy.io.votable import parse_single_table


url = 'http://nesssi.cacr.caltech.edu/cgi-bin/getcssconedbid_release2.cgi'
values = {'Name': 'Her X-1', 'DB': 'photcat', 'OUT': 'vot', 'SHORT': 'short'}

data = urllib.parse.urlencode(values)
data = data.encode('ascii') # data should be bytes
req = urllib.request.Request(url, data)

with urllib.request.urlopen(req) as response:
   the_page = response.read().decode('utf-8')
   split_page = the_page.split('\n')[-7]
   parsed = split_page[len('<font size=2> (right-mouse-click and save as to <a href='):]
   data_url= parsed.rstrip('>download</a>)')


   with urllib.request.urlopen(data_url) as output:
     vot_file = output.read().decode()
     urllib.request.urlretrieve(data_url, filename ='data_vot.xml')

table = parse_single_table('data_vot.xml')

magnitude = table.array['Mag']
ObsTime = table.array['ObsTime']

plt.scatter(ObsTime,magnitude)   # Default plot
plt.title("Magnitute against time")        # Add title/lables
plt.xlabel("Time")
plt.ylabel("Magnitude")
#plt.savefig('trial.pdf')    
plt.show()                    # show the actual plot

