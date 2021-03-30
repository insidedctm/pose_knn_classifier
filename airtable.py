import requests
import argparse

AIRTABLE_BASE_ID = 'appHPd79bznxMtc2Q'
AIRTABLE_API_KEY = 'keyfgenGdzjVNltxE'

base_endpoint = f'https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/Data%20collection?api_key={AIRTABLE_API_KEY}'

headers = {
	"Authorization": f"Bearer {AIRTABLE_API_KEY}"
}

def dump_table(maxRecords=3):
  endpoint = f'{base_endpoint}&maxRecords={maxRecords}&filterByFormula=%7BRef%7D%20%3D%20%27181%27'
  r = requests.get(endpoint, headers)

  print(dir(r))
  print(r.reason)
  json = r.json()
  print(json.keys())
  print(json)

def video_url_by_ref(ref):
  endpoint = f'{base_endpoint}&filterByFormula=%7BRef%7D%20%3D%20%27{ref}%27'
  r = requests.get(endpoint, headers)

  records = r.json()['records']
  if len(records) > 0:
    video = records[0]['fields']['Video']
    if len(video) > 0:
      print(video)
      return video[0]['url']

  print('No video url found')
  return ''

def download_video_by_ref(ref):
  url = video_url_by_ref(ref)
  if url == '':
    print('no url found')
    filename = ''
  else:
    filename = download_file(url)

  return filename

def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_filename

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--video_by_ref', default=None)
  args = parser.parse_args()
  if args.video_by_ref:
    url = video_url_by_ref(args.video_by_ref)
    print(url)
  else:
    dump_table() 
