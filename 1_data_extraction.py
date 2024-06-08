# -*- coding: utf-8 -*-
# @Time:24/04/2023 4:20 pm
# @Author:zhenyun deng
# @File:1_extract_texts_from_url.py

# step1: extract all available original_claim_url from 'data-clean-dev.with_orig.json
# step2: extract all texts from urls
# step3: normalize these extracted texts
# step4: feed them into ChatGPT => see more: extract_claims_with_chatgpt.py

import requests
import openai
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import string
import nltk
import os
from collections import Counter
import requests
import pandas as pd
import re
import unicodedata
import argparse
from bs4 import BeautifulSoup, Tag, NavigableString
from tqdm import tqdm
import trafilatura
from trafilatura.meta import reset_caches
from trafilatura.settings import DEFAULT_CONFIG

DEFAULT_CONFIG.MAX_FILE_SIZE = 50000

pattern = re.compile(r'<[^>]+>', re.S)

RULINGS = ['true', 'mostly-true', 'half-true', 'barely-true', 'false',
           'pants-fire']
RULINGS_IN_TEXT = ['true', 'mostly true', 'half true', 'mostly false', 'false',
                   'pants on fire', 'barely true']
RULING_SEC_PATTERN = "Our [R,r][uling,ating]"


def extract_all_paragraphs(paras):
    evidence = []
    for para in paras:
        if para is not None \
                and not isinstance(NavigableString, NavigableString) \
                and not para == '\n':
            child_num = len(para.find_all())
            # no child other than hyperlinks
            if para.name == 'table' or (para.name == 'div' and child_num >= 2):
                pass
            else:
                text = para.get_text(strip=True)
                if text:
                    evidence.append(unicodedata.normalize(
                        "NFKC",
                        text)
                    )
    return evidence


def get_sibling(element):
    sibling = element.next_sibling
    if sibling == "\n" or isinstance(sibling, NavigableString):
        return get_sibling(sibling)
    else:
        return sibling


def extract_texts_from_politifact_url(page_url):
    #
    r = requests.get(page_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    name = soup.find(attrs={"class": "m-statement__name"})
    # find content
    claim = soup.find('div', attrs={"class": "m-statement__quote"})
    # find the full article
    full_article = soup.find('article', attrs={"class": 'm-textblock'})
    # find when & where
    time_venue = soup.find('div', attrs={"class": "m-statement__desc"})
    time_venue = time_venue.get_text(strip=True)
    name = name.get_text(strip=True)
    claim = claim.get_text(strip=True)
    paras = full_article.find_all('p')
    full_article = extract_all_paragraphs(paras)
    anchor = soup.find('div', text=re.compile(RULING_SEC_PATTERN),
                       recursive=True)

    if not anchor:
        anchor = soup.find('strong', text=re.compile(RULING_SEC_PATTERN),
                           recursive=True)
    if not anchor:
        anchor = soup.find('p', text=re.compile(RULING_SEC_PATTERN),
                           recursive=True)
    while get_sibling(anchor) is None or get_sibling(anchor) == '\n':
        anchor = anchor.parent
    justification_para = []
    anchor = get_sibling(anchor)
    while isinstance(anchor, Tag):
        if not anchor.find('p') and not anchor.name == 'p':
            anchor = get_sibling(anchor)
            continue
        paras = anchor.find_all('p')
        if not paras:
            paras = [anchor]
        for para in paras:
            if para is not None \
                    and not isinstance(NavigableString, NavigableString) \
                    and not para == '\n':
                justification_para.append(unicodedata.normalize(
                    "NFKC",
                    para.get_text(strip=True)
                )
                )
        anchor = get_sibling(anchor)

    while justification_para[-1] != full_article[-1]:
        justification_para.pop()
    for i in range(len(justification_para)):
        full_article.pop()

    full_context = ' '.join(full_article)
    return full_article, full_context


def extract_texts_from_url_ori(url):
    response = requests.get(url)
    # response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.content, "html.parser")

    # step3: extract english_texts
    english_texts = [text for text in soup.stripped_strings if text.isascii()]
    english_texts = " ".join(english_texts)

    # step3: extract all texts
    texts = soup.get_text()
    texts = texts.strip().replace("\n", "")  # texts = texts.strip().split('\n')

    # step3:
    # string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    # fil = re.compile(u'[^0-9a-zA-Z!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')
    # texts = fil.sub(' ', texts)
    # text = ' '.join(text.split())

    return texts, english_texts


def col_avail_url(data):
    avail_orig_claim_url = []

    for sample in tqdm(data):
        original_claim_url = sample['original_claim_url']

        # step1: extract available url
        if original_claim_url is None or original_claim_url == "":
            continue

        # filter out url if url in ['youtube', '.png', '.jpg', 'type=image', '.pdf']
        if '.jpg' in original_claim_url:
            continue
        if '.png' in original_claim_url:
            continue
        if 'youtube' in original_claim_url:
            continue
        if 'type=image' in original_claim_url:
            continue
        if '.pdf' in original_claim_url:
            continue

        # step2: extract all texts from url
        try:
            response = requests.get(original_claim_url, timeout=30)
            if response.status_code == 200:
                avail_orig_claim_url.append(sample)
        except Exception as e:
            print("{} can't be accessed".format(original_claim_url))
            pass

    return avail_orig_claim_url




def text_in_facebook(url):
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.content, "html.parser")
    full_html = soup.find('html', {'class': '_9dls __fb-light-mode'})
    full_text = ''

    if full_html:
        meta_list = full_html.find_all('meta')
        for meta in meta_list:
            if meta.attrs:
                if 'property' in meta.attrs.keys():
                    if meta.attrs['property'] == 'og:title' or meta.attrs['property'] == 'og:image:alt':
                        full_text = meta.attrs['content']

    #
    if not full_text and 'archive.org' not in url:
        prefix = "https://web.archive.org/web/"  # "https://web.archive.org/"

        response = requests.get(prefix + url, timeout=30)
        soup = BeautifulSoup(response.content, "html.parser")
        full_html = soup.find('html', {'data-scribe-reduced-action-queue': 'true'})

        if full_html:
            meta_list = full_html.find_all('meta')
            for meta in meta_list:
                if meta.attrs:
                    if 'property' in meta.attrs.keys():
                        if meta.attrs['property'] == 'og:description':
                            full_text = meta.attrs['content']
        else:
            page = trafilatura.fetch_url(prefix + url, config=DEFAULT_CONFIG)
            if page:
                text_list = re.findall(r"<p>(.+?)</p>", page)
                #
                text_list_filter = []
                for text in text_list:
                    result = pattern.sub('', text)
                    text_list_filter.append(result)
                #
                full_text = ' '.join(text_list_filter)
    #
    if not full_text:
        page = trafilatura.fetch_url(url, config=DEFAULT_CONFIG)
        if page:
            text_list = re.findall(r"<p>(.+?)</p>", page)
            #
            text_list_filter = []
            for text in text_list:
                result = pattern.sub('', text)
                text_list_filter.append(result)
            #
            full_text = ' '.join(text_list_filter)
    #
    if not full_text:
        texts = soup.get_text()
        full_text = texts.strip().replace("\n", "")

    if "Facebook" == full_text or "FacebookFacebookEmail" in full_text or "FacebookNoticeYou" in full_text or \
            'English (US)' in full_text or 'English (UK)' in full_text or 'Page not found' in full_text:
        full_text = ''

    return full_text


def text_in_twitter(url):
    full_text = ''
    if 'status' in url and 'archive.org' in url:
        # full_text = extract_text_from_twitter(full_text, url)
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.content, "html.parser")
        full_html = soup.find('html', {'data-scribe-reduced-action-queue': 'true'})

        if full_html:
            meta_list = full_html.find_all('meta')
            for meta in meta_list:
                if meta.attrs:
                    if 'property' in meta.attrs.keys():
                        if meta.attrs['property'] == 'og:description':
                            full_text = meta.attrs['content']
        else:
            texts = soup.get_text()
            full_text = texts.strip().replace("\n", "")

    if 'status' in url and 'archive.org' not in url:
        prefix = "https://web.archive.org/web/"  # "https://web.archive.org/"

        response = requests.get(prefix + url, timeout=30)
        soup = BeautifulSoup(response.content, "html.parser")
        full_html = soup.find('html', {'data-scribe-reduced-action-queue': 'true'})

        if full_html:
            meta_list = full_html.find_all('meta')
            for meta in meta_list:
                if meta.attrs:
                    if 'property' in meta.attrs.keys():
                        if meta.attrs['property'] == 'og:description':
                            full_text = meta.attrs['content']
        else:
            page = trafilatura.fetch_url(prefix + url, config=DEFAULT_CONFIG)
            tmp_contents = page.split('<meta data-rh="true" ')

            if tmp_contents:
                for cont in tmp_contents:
                    if 'property="og:description"' in cont:
                        cont_list = cont.split('property="og:description"')
                        for _cont in cont_list:
                            if "content" in _cont:
                                full_text = _cont[10:-3]
            else:
                texts = soup.get_text()
                full_text = texts.strip().replace("\n", "")

    if not full_text:
        page = trafilatura.fetch_url(url, config=DEFAULT_CONFIG)
        if page:
            text_list = re.findall(r"<p>(.+?)</p>", page)
            #
            text_list_filter = []
            for text in text_list:
                result = pattern.sub('', text)
                text_list_filter.append(result)
            #
            full_text = ' '.join(text_list_filter)

    #
    if not full_text:
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.content, "html.parser")
        texts = soup.get_text()
        full_text = texts.strip().replace("\n", "")

    if 'JavaScript is not available' in full_text:
        full_text = ''

    return full_text


def text_in_twitter_ori(url):
    full_text = ''
    if 'status' in url and 'archive.org' in url:
        # prefix = "https://web.archive.org/web/"     # "https://web.archive.org/"
        # page = trafilatura.fetch_url(prefix + url, config=DEFAULT_CONFIG)
        page = trafilatura.fetch_url(url, config=DEFAULT_CONFIG)
        tmp_contents = page.split('<meta data-rh="true" ')
        for cont in tmp_contents:
            if 'property="og:description"' in cont:
                cont_list = cont.split('property="og:description"')
                for _cont in cont_list:
                    if "content" in _cont:
                        full_text = _cont[10:-3]
    else:
        if 'archive.org' in url and 'twitter' in url:
            prefix = "https://"
            url_list = url.split(prefix)
            for _url in url_list:
                if 'twitter' in _url:
                    url = prefix + _url

        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.content, "html.parser")
        full_html = soup.find('html', {'data-scribe-reduced-action-queue': 'true'})

        if full_html:
            meta_list = full_html.find_all('meta')
            for meta in meta_list:
                if meta.attrs:
                    if 'property' in meta.attrs.keys():
                        if meta.attrs['property'] == 'og:description':
                            full_text = meta.attrs['content']
        else:
            texts = soup.get_text()
            full_text = texts.strip().replace("\n", "")

    return full_text


def text_in_permacc(url):
    full_text = ''
    if 'archive.org' in url:
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.content, "html.parser")
        meta_list = soup.find_all('meta')

        for meta in meta_list:
            if meta.attrs:
                if 'property' in meta.attrs.keys():
                    if meta.attrs['property'] == 'og:description':
                        full_text = meta.attrs['content']
        if not full_text:
            texts = soup.get_text()
            full_text = texts.strip().replace("\n", "")
    else:
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.content, "html.parser")
        full_html = soup.find('input', {'class': 'tray-detail-entry'})
        ori_url = full_html.attrs['value']

        if "facebook" in ori_url:
            full_text = text_in_facebook(ori_url)
        elif "twitter" in ori_url:
            full_text = text_in_twitter(ori_url)
        else:
            page = trafilatura.fetch_url(url, config=DEFAULT_CONFIG)
            if page:
                texts = trafilatura.extract(page, config=DEFAULT_CONFIG)
                if texts is None:
                    return ''
                full_text = texts.strip().replace("\n", "")
            else:
                texts = soup.get_text()
                full_text = texts.strip().replace("\n", "")

    if "Perma.cc" in full_text:
        full_text = ''

    return full_text


def text_in_president(url):
    page = trafilatura.fetch_url(url, config=DEFAULT_CONFIG)
    full_text = ''

    if page:
        full_text = trafilatura.extract(page, config=DEFAULT_CONFIG)
        # if re.findall(r"<p dir=\"ltr\">(.+?)</p>", page):
        #     text_list = re.findall(r"<p dir=\"ltr\">(.+?)</p>", page)
        #     full_text = ' '.join(text_list)
        # else:
        #     text_list = re.findall(r"<p>(.+?)</p>", page)
        #     full_text = ' '.join(text_list)

        # text_list = re.findall(r"<p>(.+?)</p>", page)
        # full_text = ' '.join(text_list)
        # text_list = re.findall(r"<p(.+?)>(.+?)</p>", page)
        # texts = []
        # for t0, t1 in text_list:
        #     if t0[0] == '>':
        #         text = t0[1:] + t1
        #     else:
        #         text = t1
        #     texts.append(text)
        # full_text = ' '.join(texts)
    else:
        full_text = ''

    return full_text


def text_in_cspan(url):
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.content, "html.parser")
    meta_list = soup.find_all('meta')
    full_text = ''

    for meta in meta_list:
        if meta.attrs:
            if 'property' in meta.attrs.keys():
                if meta.attrs['property'] == 'og:description':
                    full_text = meta.attrs['content']

    return full_text


def text_in_axios(url):
    page = trafilatura.fetch_url(url, config=DEFAULT_CONFIG)

    if page:
        # full_text = trafilatura.extract(page, config=DEFAULT_CONFIG)
        text_list = re.findall(r"<p>(.+?)</p>", page)
        #
        text_list_filter = []
        for text in text_list:
            result = pattern.sub('', text)
            text_list_filter.append(result)
        #
        full_text = ' '.join(text_list_filter)
    else:
        full_text = ''

    return full_text


def text_in_general_web(url):
    """
    www.president.go.ke # text_in_president(url)
    www.aljazeera.com
    www.gov.za
    www.news24.com
    www.punchng.com
    abcnews.go.com/
    cnn.com
    www.washingtonpost.com
    www.cbsnews.com
    www.telegraph.co.uk
    www.foxnews.com
    www.nbcnews.com
    channelstv.com
    """
    page = trafilatura.fetch_url(url, config=DEFAULT_CONFIG)

    if page:
        full_text = trafilatura.extract(page, config=DEFAULT_CONFIG)
    else:
        full_text = ''

    return full_text


def extract_fulltext_from_avail_url(datas):
    # datas = json.load(open(file_path, 'r'))

    fulltext_data = []
    for data in tqdm(datas):
        url = data['original_claim_url']
        try:
            if "facebook.com" in url:
                texts = text_in_facebook(url)
            elif "twitter.com" in url:
                # continue      # for debugging
                texts = text_in_twitter(url)
            elif "perma.cc" in url:
                texts = text_in_permacc(url)
            elif "www.c-span.org" in url:
                texts = text_in_cspan(url)
            elif "www.axios.com" in url:
                texts = text_in_axios(url)
            elif ".pdf" in url:
                continue
            else:
                texts = text_in_general_web(url)

            if texts:
                data['fulltext'] = texts
                fulltext_data.append(data)
        except Exception as e:
            print("{} can't be accessed".format(url))
            pass

    return fulltext_data



def convert_json_xlsx(save_path, file_path):
    datas = json.load(open(file_path, 'r'))

    pre_fullname = file_path.split('.json')[0]
    url_list, claim_list, fulltext_list = [], [], []

    for data in tqdm(datas):
        url_list.append(data['original_claim_url'])
        claim_list.append(data['claim'])
        fulltext_list.append(data['fulltext'])

    data_csv = pd.DataFrame({"url": url_list, "claim": claim_list, "fulltext": fulltext_list})
    data_csv.to_excel('{}.xlsx'.format(pre_fullname), sheet_name='sheet1', index=False)
    print("json -> xlsx finished!")


if __name__ == '__main__':
    filename = ['train', 'dev', 'test']  # train, dev, test
    data_path = 'all_data'
    save_path = 'all_data/averitec_data'

    all_avail_samples = []
    for _fn in filename:
        input_data = '{}/{}.json'.format(save_path, _fn)
        data = json.load(open(input_data, 'r'))

        # step 1: collect available original_url from AVeriTeC
        avail_url = col_avail_url(data)

        # step 2: extract texts from available original_url
        fulltext_in_url = extract_fulltext_from_avail_url(avail_url)

        all_avail_samples.extend(fulltext_in_url)

    with open('{}/1_all_available_url_fulltext.json'.format(data_path), 'w', encoding='utf-8') as f:
        json.dump(all_avail_samples, f, ensure_ascii=False)





