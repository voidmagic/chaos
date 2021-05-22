import sys
import pathlib


low = ['ha', 'am', 'kk', 'ps', 'ug', 'be', 'te', 'fy', 'zu', 'se', 'oc', 'ky', 'mr', 'li', 'my', 'ig', 'gd', 'yi', 'kn', 'or', 'tk']
mid = ['ml', 'ur', 'mg', 'hi', 'gl', 'nn', 'xh', 'ne', 'ka', 'eo', 'gu', 'ga', 'cy', 'af', 'sh', 'az', 'ta', 'tg', 'rw', 'uz', 'br', 'ku', 'nb', 'as', 'km', 'pa', 'wa', 'tt']
high = ['zh', 'vi', 'uk', 'tr', 'th', 'sv', 'sr', 'sq', 'sl', 'sk', 'ru', 'ro', 'pt', 'pl', 'no', 'nl', 'mt', 'ms', 'mk', 'lv', 'lt', 'ko', 'ja', 'it', 'is', 'id', 'hu', 'hr', 'he', 'fr', 'fi', 'fa', 'eu', 'et', 'es', 'el', 'de', 'da', 'cs', 'ca', 'bs', 'bn', 'bg', 'ar', 'si']


path = pathlib.Path(sys.argv[1])

results = {}

for folder in path.glob('*'):
    if not folder.is_dir():
        continue
    if 'result' in folder.name:
        # multilingual model
        lang = folder.name.split('.')[-1]
        for result_file in folder.glob('*txt'):
            bleu = result_file.read_text().strip().split('\n')[-1].split()[6].strip(',')
            results[lang] = float(bleu)
    else:
        # bilingual model
        lang = folder.name.split('-')
        lang.remove('en')
        lang = lang[0]
        for result_folder in folder.glob('*result*'):
            for result_file in result_folder.glob('*txt'):
                bleu = result_file.read_text().strip().split('\n')[-1].split()[6].strip(',')
                results[lang] = float(bleu)


print('All:', sum(results.values()) / len(results))
low_bleu = {key: value for key, value in results.items() if key in low}
mid_bleu = {key: value for key, value in results.items() if key in mid}
high_bleu = {key: value for key, value in results.items() if key in high}
print('Low:', sum(low_bleu.values()) / len(low_bleu))
print('Mid:', sum(mid_bleu.values()) / len(mid_bleu))
print('High:', sum(high_bleu.values()) / len(high_bleu))
