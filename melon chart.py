import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
import time
import csv

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
    if answer[0] == 'happy':
        if ', ' in genre[2].text:
            if '댄스' in genre[2].text:
                genre1.append('댄스')
            elif '성인가요/트로트' in genre[2].text:
                genre1.append('성인가요/트로트')
            elif '랩/힙합' in genre[2].text:
                genre1.append('랩/힙합')
        else:
            genre1.append(genre[2].text)
    if answer[0] == 'sad':
        if ', ' in genre[2].text:
            if '발라드' in genre[2].text:
                genre1.append('발라드')
            elif '재즈' in genre[2].text:
                genre1.append('재즈')
        else:
            genre1.append(genre[2].text)
    if answer[0] == 'angry':
        if ', ' in genre[2].text:
            if '발라드' in genre[2].text:
                genre1.append('발라드')
            elif '록/메탈' in genre[2].text:
                genre1.append('록/메탈')
        else:
            genre1.append(genre[2].text)
    if answer[0] == 'disgust':
        if ', ' in genre[2].text:
            if '댄스' in genre[2].text:
                genre1.append('댄스')
            elif '성인가요/트로트' in genre[2].text:
                genre1.append('성인가요/트로트')
            elif '랩/힙합' in genre[2].text:
                genre1.append('랩/힙합')
        else:
            genre1.append(genre[2].text)
    if answer[0] == 'calm':
        if ', ' in genre[2].text:
            if '댄스' in genre[2].text:
                genre1.append('댄스')
            elif '발라드' in genre[2].text:
                genre1.append('발라드')
            elif 'POP' in genre[2].text:
                genre1.append('POP')
        else:
            genre1.append(genre[2].text)
    if answer[0] == 'neutral':
        if ', ' in genre[2].text:
            if '댄스' in genre[2].text:
                genre1.append('댄스')
            elif '발라드' in genre[2].text:
                genre1.append('발라드')
            elif 'POP' in genre[2].text:
                genre1.append('POP')
        else:
            genre1.append(genre[2].text)
    if answer[0] == 'surprised':
        if ', ' in genre[2].text:
            if 'POP' in genre[2].text:
                genre1.append('POP')
            elif '랩/힙합' in genre[2].text:
                genre1.append('랩/힙합')
        else:
            genre1.append(genre[2].text)
    if answer[0] == 'fearful':
        if ', ' in genre[2].text:
            if '포크/블루스' in genre[2].text:
                genre1.append('포크/블루스')
        else:
            genre1.append(genre[2].text)

j = 0

for i in lst50:
    temp = []
    #temp.append(i.select_one('.rank').text)  #순위 가져오기
    temp.append(i.select_one('.ellipsis.rank01').a.text) #제목 가져오기
    temp.append(i.select_one('.ellipsis.rank02').a.text) #아티스트 가져오기
    #temp.append(i.select_one('.ellipsis.rank03').a.text) #앨범 가져오기
    temp.append(genre1[j])
    melonList.append(temp)
    j+=1

#print(melonList)  #주석을 풀고 여기까지 실행하면 작동 여부 확인 됨

#####담은 정보를 파일로 저장하기#####
with open('melon100_utf8.csv', 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    #writer.writerow(['순위', '곡명', '아티스트', '앨범', '장르'])
    #writer.writerow(['곡명', '장르'])
    writer.writerows(melonList)

#with open('melon100_cp949.csv', 'w', encoding='cp949', newline='') as f:
  #  writer = csv.writer(f)
  #  writer.writerow(['순위', '아티스트', '곡명', '앨범'])
  #  writer.writerows(melonList)

print("완료!")
