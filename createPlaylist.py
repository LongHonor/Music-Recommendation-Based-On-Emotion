#!/usr/bin/python

import httplib2
import os
import sys
import csv
from datetime import datetime

from apiclient.discovery import build
from apiclient.errors import HttpError
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow
from tkinter import messagebox



# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret. You can acquire an OAuth 2.0 client ID and client secret from
# the Google API Console at
# https://console.developers.google.com/.
# Please ensure that you have enabled the YouTube Data API for your project.
# For more information about using OAuth2 to access the YouTube Data API, see:
#   https://developers.google.com/youtube/v3/guides/authentication
# For more information about the client_secrets.json file format, see:
#   https://developers.google.com/api-client-library/python/guide/aaa_client_secrets
CLIENT_SECRETS_FILE = "client_secrets.json"

# This variable defines a message to display if the CLIENT_SECRETS_FILE is
# missing.
MISSING_CLIENT_SECRETS_MESSAGE = """
WARNING: Please configure OAuth 2.0
To make this sample run you will need to populate the client_secrets.json file
found at:
   %s
with information from the API Console
https://console.developers.google.com/
For more information about the client_secrets.json file format, please visit:
https://developers.google.com/api-client-library/python/guide/aaa_client_secrets
""" % os.path.abspath(os.path.join(os.path.dirname(__file__),
                                   CLIENT_SECRETS_FILE))

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account.
YOUTUBE_READ_WRITE_SCOPE = "https://www.googleapis.com/auth/youtube"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

flow = flow_from_clientsecrets(CLIENT_SECRETS_FILE,
  message=MISSING_CLIENT_SECRETS_MESSAGE,
  scope=YOUTUBE_READ_WRITE_SCOPE)

storage = Storage("%s-oauth2.json" % sys.argv[0])
credentials = storage.get()

if credentials is None or credentials.invalid:
  flags = argparser.parse_args()
  credentials = run_flow(flow, storage, flags)

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
  http=credentials.authorize(httplib2.Http()))

ytlinkfront = "https://www.youtube.com/playlist?list="

class playlistCreater:
  def __init__(self):
    self.playlink = ""

  # 플레이리스트 만들기
  def createPlayList(self, plTitle, plDescription):
    playlists_insert_response = youtube.playlists().insert(
      part = "snippet, status",
      body = dict(
        snippet=dict(
          title = plTitle,
          description = plDescription
        ),
        status=dict(
          privacyStatus="public"
        )
      )
    ).execute()
    
    return playlists_insert_response["id"]

  #플레이리스트에 동영상 추가
  def insertionToPlayList(self, playlist, vid):
    playlistsItems_insert_request = youtube.playlistItems().insert(
      part = "snippet",
      body = {
        'snippet': {
          'playlistId' : playlist,
          'resourceId' : {
            'kind' : 'youtube#video',
            'videoId' : vid
          }
        }
      }
    )

    playlistsItems_insert_response = playlistsItems_insert_request.execute()

  #검색해서 vid 찾기
  def getVideoIdBySearch(self, string):
    search_request = youtube.search().list(
      q = string,
      order = "relevance",
      part = "id, snippet",
      maxResults = 1
    )
    search_response = search_request.execute()
    return search_response['items'][0]['id']['videoId']

  def resultingPlaylist(self,tmp):
    title = datetime.today().strftime("%Y%m%d")
    description = title + "의 플레이리스트"
    plid = self.createPlayList(title, description)
    self.playlink = ytlinkfront + plid

    file = open("classified_melonList.csv", 'r', encoding = 'utf-8-sig')
    rdr = csv.reader(file)
    for line in rdr:
        searchWord = line[0] + ' ' + line[1]
        vid = self.getVideoIdBySearch(searchWord)
        self.insertionToPlayList(plid, vid)
    file.close()
   
    print(self.playlink)
    messagebox.showinfo("작업완료","플레이리스트 생성 완료")
