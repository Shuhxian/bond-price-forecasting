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

def inflation_rate_extract():
  !pip install selenium -q
  !apt-get update # to update ubuntu to correctly run apt install
  !apt install chromium-chromedriver
  !cp /usr/lib/chromium-browser/chromedriver /usr/bin
  import sys
  sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
  from selenium import webdriver
  from selenium.webdriver.support.ui import WebDriverWait
  from selenium.webdriver.support import expected_conditions as EC
  from selenium.webdriver.common.by import By
  from selenium.webdriver.common.action_chains import ActionChains
  from selenium.webdriver.support.ui import Select
  import datetime
  from dateutil.relativedelta import relativedelta

  chrome_options = webdriver.ChromeOptions()
  chrome_options.add_argument('--headless')
  chrome_options.add_argument('--no-sandbox')
  chrome_options.add_argument('--disable-dev-shm-usage')
  chrome_options.add_argument("start-maximized")
  chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
  chrome_options.add_experimental_option('useAutomationExtension', False)

  wd = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
  wd.get("https://www.fxempire.com/macro/malaysia/inflation-rate")

  #Most recent three years
  wd.find_element_by_xpath('//div[@class=" css-16ycfp3"]').click()
  wd.find_element_by_xpath("//div[@class=' css-14rzzno-menu']//*[name()='div']").click()

  wd.execute_script("return arguments[0].scrollIntoView(true);", WebDriverWait(wd, 20).until(EC.visibility_of_element_located((By.XPATH, "//div[@class='recharts-wrapper']"))))
  elements = WebDriverWait(wd, 20).until(EC.visibility_of_all_elements_located((By.XPATH, "//div[@class='recharts-wrapper']//*[name()='svg']//*[name()='g' and @class='recharts-layer recharts-bar']//*[name()='g']//*[name()='g']//*[name()='g']")))
  ir=[]
  for element in elements:
    ActionChains(wd).move_to_element(element).perform()
    mouseover = WebDriverWait(wd, 5).until(EC.visibility_of_element_located((By.XPATH, "//div[@class='recharts-wrapper']")))
    ir.append(float(mouseover.text.strip().replace("%","")[-6:].replace("\n","")))
  current_date=datetime.date.today()-relativedelta(days=datetime.date.today().day)
  inflation_dict={}
  for i in range(len(ir)-1,-1,-1):
    inflation_dict[str(current_date)]=ir[i]
    current_date-=relativedelta(days=current_date.day)
  inflation_dict['2021-06-30']=3.4
  inflation_dict['2021-05-31']=4.4
  return inflation_dict

def assign_ir(value_date):
  for date, ir in inflation_dict.items(): 
    date = dateutil.parser.parse(date) 
    if value_date >= date:
      return ir
