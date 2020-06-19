"""
Ph21 Assignment 1 part 1
Created By Kyriacos Xanthos

"""

import urllib.request
import urllib.parse
import numpy as np
import matplotlib.pyplot as plt


url = 'http://nesssi.cacr.caltech.edu/cgi-bin/getcssconedbid_release2.cgi'
values = {'Name': 'Her X-1', 'DB': 'photcat', 'OUT': 'csv', 'SHORT': 'short'}

data = urllib.parse.urlencode(values)
data = data.encode('ascii') # data should be bytes
req = urllib.request.Request(url, data)


with urllib.request.urlopen(req) as response:
   the_page = response.read().decode('utf-8')
   split_page = the_page.split('\n')[-3]
   parsed = split_page[len('<font size=2> (right-mouse-click and save as to <a href='):]
   data_url= parsed.rstrip('>download</a>)')
   
   with urllib.request.urlopen(data_url) as output:
      csv_file = output.read().decode()
      #print(csv_file)
      
"""
I used this to export the CSV file. After one export there was no need to keep it in my code

f = open('csvfile.csv','w')
f.write(csv_file) #Give your csv text here.
## Python will convert \n to os.linesep
f.close()
"""

data = np.genfromtxt('csvfile.csv', comments='#', delimiter=',', skip_header = 1)

magnitude  = data[:,1]


MJD = data[:,5]

plt.scatter(MJD,magnitude)   # Default plot
plt.title("Magnitute against time")        # Add title/lables
plt.xlabel("Time")
plt.ylabel("Magnitude")
#plt.savefig('trial1.pdf')    
plt.show()                    # show the actual plot


