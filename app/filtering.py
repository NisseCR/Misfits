import pandas as pd
import re

# read in the raw data
def _read_data() -> pd.DataFrame:
    return pd.read_csv('./data/cleaned.csv')


# prepare the list of words to filter by
filtering_words = [r'^chin', r'^sino', 'mainland', 'PRC', r'^ccp$', 'Zhongguo']
politics = ['belt and road initiative', r'^BRI$' ,'Xi Jinping', 'Hu Jintao', 'Jiang Zemin', 'Yang Shangkun', 'Li Xiannian',
            'Liu Shaoqi', 'Mao Zedong', r'^Mao$', 'SASAC', 'FOCAC', "People's Liberation Party", 'CADF', 'CAETC']
ambassadors = ['zhou pingjian', 'cui jianchun', 'sun baohong', 'wang shiting', 'lu kun']
places_in_china = ['beijing', 'shanghai', 'hong kong', 'guangzhou','chengdu', 'chongqing', 'shenzhen', 'tianjin', 'dongguan',
                'hangzhou', 'nanjing', 'xian', 'qingdao', 'shenyang','foshan','harbin','macau', 'guangdong', 'yunnan',
                   'hunan', 'wuhan', 'sichuan','hubei', 'tibet', 'xinjiang', 'Zhejiang', 'Fujian', 'Shandong']
major_companies_in_Nigeria = ['sinopec', 'cnpc', 'sepco', 'ccecc', 'cscec', 'cnoon', 'huawei', 'ztc']
# removed company names containing china, shanghai, since it's already picked up opon
top_50_companies = ['tencent', 'icbc', 'kweichow moutai','petrochina', 'alibaba', '^catl$', 'pinduoduo', 'cm bank',
                    'cnooc', 'xiaomi', 'ping an insurance', 'meituan', 'byd', 'sinopec', 'china telecom', 'midea',
                    'wuliangye yibin','bank of communications', 'netease', 'zijin mining', 'industrial bank',
                    'foxconn industrial internet', 'jingdong mall', 'citic securities', 'smic', 'nongfu spring',
                    'east money information', 'citic bank', 'trip.com','luxshare precision', 'mindrey',
                    "the people’s insurance company", 'jiangsu hengrui medicine', 'gree electric appliances',
                    'hikvision', 'haier', 'foshan haitian flavouring and food', 'citic limited', 'ping an bank']
other = ['uighurs', 'confucius institute', 'mandarin', 'cantonese', 'silk road', 'forbidden city', 'great wall',
         'wechat', 'weibo',  'baidu', 'xinhua news agency', 'CCTV', 'Tiananmen Square', 'Shaolin', 'terracotta army']

filtering_words.extend(ambassadors)
filtering_words.extend(places_in_china)
filtering_words.extend(major_companies_in_Nigeria)
filtering_words.extend(top_50_companies)
filtering_words.extend(other)
set_keywords = set(filtering_words)

# funtion filtering
def _china_filter(df: pd.DataFrame, minimal_text_mention, minimal_headline_mention) -> pd.DataFrame:
    df = df.reset_index()
    col_text_mentions = []
    col_headline_mentions = []
    for index, row in df.iterrows():
        china_mention_text = []
        china_mention_headline = []
        for word in set_keywords:
            china_mention_text.extend(re.findall(word, str(row['text']), re.IGNORECASE))
            china_mention_headline.extend(re.findall(word, str(row['headline']), re.IGNORECASE))
        col_text_mentions.append(len(china_mention_text))
        col_headline_mentions.append(len(china_mention_headline))
    df['china_mention_text'] = col_text_mentions
    df['china_mention_headline'] = col_headline_mentions
    df = df[(df['china_mention_text'] >= minimal_text_mention) & (df['china_mention_headline'] >= minimal_headline_mention)]
    return df


def length_filter(df: pd.DataFrame, min_word_count, max_word_count):
    df = df[df['word_count'] >= min_word_count]
    df = df[df['word_count'] <= max_word_count]
    return df


def _write_data(df: pd.DataFrame) -> None:
    df.to_csv('./data/filtered.csv', index=False)


def preprocess() -> None:
    df = _read_data()
    df = _china_filter(df, 3, 1)
    df = length_filter(df, 100, 1000)
    _write_data(df)





