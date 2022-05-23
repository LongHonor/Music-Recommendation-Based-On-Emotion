import httplib2
import os
import sys

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow

from playlistsample import CLIENT_SECRETS_FILE, MISSING_CLIENT_SECRETS_MESSAGE, YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, YOUTUBE_READ_WRITE_SCOPE

CLIENT_SECRETS_FILE = "client_secrets.json"

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

youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,http=credentials.authorize(httplib2.Http()) )

#api_key = 'AIzaSyAN7NlWIhMOHLZnqlCH9x95Mgz9_zjH-0c'

#service build
#youtube = build('youtube', 'v3', developerKey=api_key)

#검색어 검색해서 videoId 받아오기
def getVideoIdBySearch(string):
  search_request = youtube.search().list(
    q = string,
    order = "relevance",
    part = "id, snippet",
    maxResults = 1
  )
  search_response = search_request.execute()
  return search_response['items'][0]['id']['videoId']


#플레이리스트에 동영상 추가하기
def insertionToPlayList(playlist, vid):
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
  print(playlistsItems_insert_response["id"])

insertionToPlayList('PL4uvI6mYHX6BsHvNroS38tIkQcZkt-1NF','Kr5gznNDLjI')

#플레이리스트 만들기
def createPlayList(plTitle, plDescription):
  playlists_insert_request = youtube.playlists().insert(
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
  )
  playlists_insert_response = playlists_insert_request.execute()


""" 플레이리스트 안의 영상 정보 가져오기
pl_request = youtube.playlistItems().list(
  part = 'contentDetails',
  playlistId = 'PLt93xJuzEMVYIbcp7Vv1BiPMD6x5_Achp'
)

pl_response = pl_request.execute()

vid_ids = []
for item in pl_response['items']:
  vid_ids.append(item['contentDetails']['videoId'])

vid_request = youtube.videos().list(
  part = "snippet, statistics",
  id=','.join(vid_ids)
)

vid_response = vid_request.execute()

for item in vid_response['items']:
    print('------------')
    print("제목: ", item['snippet']['title'])
    print("설명: ",item['snippet']['description'])
    print('------------')
"""