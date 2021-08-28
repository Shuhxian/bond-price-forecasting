def opr_extract():
  import requests
  from bs4 import BeautifulSoup
  URL = "https://www.bnm.gov.my/opr-decision-and-statement"
  page = requests.get(URL)
  soup = BeautifulSoup(page.content,'lxml')
  content=soup.find_all('td')[4:]
  rates=[]
  dates=[]
  for i in range(len(content)):
    if i%4==0:
      dates.append(str(content[i]).replace("<td>","").replace("</td>",""))
    elif i%2==0:
      rates.append(float(str(content[i]).replace("<td>\n\t\t","").replace("\n\t</td>","")))
    else:
      continue
  i=0
  while i in range(len(rates)-1):
    if rates[i] == rates[i+1]:
        del rates[i]
        del dates[i]
    else:
        i += 1
  OPR_movement={}
  for i in range(len(rates)):
    OPR_movement[dates[i]]=rates[i]
  return OPR_movement
