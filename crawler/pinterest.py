import os, sys, time
import requests
import json
import bs4 # beautifulSoupe4

def search(query, num_pins):

    # First access
    url     = 'https://www.pinterest.jp/search/pins/'
    headers = {
        'connection': 'keep-alive'
    }

    search_response = requests.get(url, params={'q':query}, headers=headers, stream=False)
    soup            = bs4.BeautifulSoup(search_response.text.replace('\n',''), 'html5lib')

    data_json_string = soup.find('script', type='application/json') # extract json string
    data_json        = json.loads(data_json_string.string) # convert into dictionary type variable
    results          = data_json['tree']['children'][0]['data']['results']
#    results          = data_json['resouceDataCache'][0]['children'][0]['data']['results']

    image_info_list  = []
    for r in results:
        image_info = {}
        image_info['description'] = r['description']
        image_info['link']        = r['link']
        image_info['image_url']   = r['images']['orig']['url']
        image_info['id']          = r['id']
        image_info_list.append(image_info)


    # Second or later access to load additional pins that are responded as a JSON string
    url             = 'https://www.pinterest.jp/resource/BaseSearchResource/get/'
    bookmarks       = data_json['resourceDataCache'][0]['resource']['options']['bookmarks']
    experiment_hash = data_json['context']['triggerable_experiments_hash']
    last_cookies    = search_response.cookies

    while len(image_info_list) < num_pins:

        ## Preparing parameters, headers and cookies for the "get" request
        params = {
            'source_url':'/search/pins/?q={}'.format(query),
            'data':json.dumps({
                'options':{
                    'bookmarks':bookmarks,
                    'query':query,
                    'scope':'pins',
                    'page_size':25,
                    'field_set_key':'unauth_react'
                },
                'context':{}}),
            '_':str(int(time.time())*10*10*10)
        }

        headers = {
            'Host':'www.pinterest.jp',
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:43.0) Gecko/20100101 Firefox/43.0',
            'Accept-Language':'ja,en-US;q=0.7,en;q=0.3',
            'X-Pinterest-AppState': 'background',
            'X-Pinterest-ExperimentHash': experiment_hash,
            'X-NEW-APP':'1',
            'X-APP-VERSION':'9b11f84',
            'X-Requested-With':'XMLHttpRequest',
            'Referer':'https://www.pinterest.jp',
            'cookie':json.dumps({
                '_auth':dict(last_cookies)['_auth'],
                'csrftoken':dict(last_cookies)['csrftoken'],
                '_pinterest_sess':dict(last_cookies)['_pinterest_sess']}),
            'connection':'keep-alive'
        }

        cookies = {
            '_auth':dict(last_cookies)['_auth'],
            'csrftoken':dict(last_cookies)['csrftoken'],
            '_pinterest_sess':dict(last_cookies)['_pinterest_sess'],
            'bei':'False',
            'logged_out':'True',
            'fba':'True',
            'sessionFunelEventLogged':'1'
        }

        search_response = requests.get(url, cookies=cookies, params=params, headers=headers, stream=False)
        data_json       = json.loads(search_response.text)
        results         = data_json['resource_response']['data']['results']

        bookmarks       = data_json['resource']['options']['bookmarks']
        experiment_hash = data_json['client_context']['triggerable_experiments_hash']
        last_cookies    = search_response.cookies

        for r in results:
            image_info = {}
            image_info['description'] = r['description']
            image_info['link']        = r['link']
            image_info['image_url']   = r['images']['orig']['url']
            image_info['id']          = r['id']
            image_info_list.append(image_info)

    return image_info_list


def main(argv):
    if (len(argv) < 2):
        print('Usage: python3 ' + argv[0] + ' search <number>')
        exit(1)
    keyword  = argv[1] # keyword you want to search
    num_pins = 1 # Number of pins searched

    if (len(argv) == 3):
        num_pins = int(argv[2])

    image_info_list = search(keyword, num_pins)

    print(json.dumps(image_info_list))

if __name__ == '__main__':
    main(sys.argv)

