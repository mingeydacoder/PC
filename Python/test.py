# -*- coding: UTF-8 -*-
import requests

url = 'https://opendata.cwa.gov.tw/fileapi/v1/opendataapi/O-A0001-001?Authorization=CWA-AC28087E-1FAE-4F05-8EF5-374D1DCE3E11&downloadType=WEB&format=JSON'
data = requests.get(url)
data_json = data.json()
location = data_json['cwaopendata']['dataset']['Station']
count = 0
for i in location:
  name = i['StationName']                    # 測站地點
  city = i['GeoInfo']['CountyName']  # 城市
  area = i['GeoInfo']['TownName']  # 行政區
  temp = i['WeatherElement']['AirTemperature']                     # 氣溫

  #print(city, area, name, f'{temp} 度')
  if count+1 == 274:
    print(count+1,name,city,area,temp)
  count += 1