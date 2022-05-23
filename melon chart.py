import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import csv
from selenium import webdriver


#####웹 사이트 정보 해석 및 읽어오기#####
hdr = { 'User-Agent' : 'Mozilla/5.0' }
url = 'https://www.melon.com/chart/index.htm'

req = urllib.request.Request(url, headers=hdr)
html = urllib.request.urlopen(req).read()
soup = BeautifulSoup(html, 'html.parser')

#####웹 정보 가져와 담기#####
lst50 = soup.select('.lst50, .lst100') # .xxx 는 Class 표기

melonList = []

cell_line = []
genre1 = []

for i in lst50:
    cell_line.append(i['data-song-no'])

for j in range(0,100):
    url2 = 'https://www.melon.com/song/detail.htm?songId=' + cell_line[j]
    req2 = urllib.request.Request(url2, headers=hdr)
    html2 = urllib.request.urlopen(req2).read()
    soup2 = BeautifulSoup(html2, 'html.parser')
    attr = soup2.select_one('dl')
    genre = attr.select('dd')
    genre1.append(genre[2].text)

j = 0

for i in lst50:
    temp = []
    temp.append(i.select_one('.rank').text)  #순위 가져오기
    temp.append(i.select_one('.ellipsis.rank01').a.text) #제목 가져오기
    temp.append(i.select_one('.ellipsis.rank02').a.text) #아티스트 가져오기
    temp.append(i.select_one('.ellipsis.rank03').a.text) #곡명
    temp.append(genre1[j])
    melonList.append(temp)
    j+=1

#for j in range(0,100):
#    url2 = 'https://www.melon.com/song/detail.htm?songId=' + cell_line[j]
#    req2 = urllib.request.Request(url2, headers=hdr)
#    html2 = urllib.request.urlopen(req2).read()
#    soup2 = BeautifulSoup(html2, 'html.parser')
#    attr = soup2.select_one('dl')
#    genre = attr.select('dd')
#    print(genre[2].text)

    

#for i in lst50:
    #links = soup.find(class_='lst50').find_all('a')
    #for j in links:
        #href = j.attrs['href']
        #cell_line.append(href)

#print(cell_line)

#print(melonList)  #주석을 풀고 여기까지 실행하면 작동 여부 확인 됨

#####담은 정보를 파일로 저장하기#####
with open('melon100_utf8.csv', 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['순위', '아티스트', '곡명', '앨범', '장르'])
    writer.writerows(melonList)

#with open('melon100_cp949.csv', 'w', encoding='cp949', newline='') as f:
  #  writer = csv.writer(f)
  #  writer.writerow(['순위', '아티스트', '곡명', '앨범'])
  #  writer.writerows(melonList)

print("완료!")
