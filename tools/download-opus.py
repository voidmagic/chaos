

def parse(url):
    # format https://object.pouta.csc.fi/OPUS-GNOME/v1/moses/es-zh_CN.txt.zip
    prefix = url[:33]
    postfix = url[33:]
    part, version, lang = postfix.split('/')
    _, lang, _ = lang.split('.')
    url = prefix + part + '/' + version + '/moses/' + lang + '.txt.zip'
    cmd = 'wget -O {}-{}.zip {}'.format(part, lang, url)
    cmd = 'unzip {}-{}.zip -d {}-{}'.format(part, lang, part, lang)
    src, tgt = lang.split('-')
    cmd = '{}-{}/{}.{}.{} \\'.format(part, lang, part, lang, src)
    return cmd


def main():
    urls = get_urls().split("\n")
    urls = [url.strip() for url in urls]
    urls = [parse(url) for url in urls]
    print("\n".join(urls))


def get_urls():
    return """https://object.pouta.csc.fi/OPUS-EUbookshop/v2/EUbookshop.es-zh.es 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es-zh_CN.es 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_AR-zh_CN.es_AR 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_CL-zh_CN.es_CL 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_CO-zh_CN.es_CO 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_CR-zh_CN.es_CR 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_DO-zh_CN.es_DO 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_EC-zh_CN.es_EC 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_ES-zh_CN.es_ES 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_GT-zh_CN.es_GT 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_HN-zh_CN.es_HN 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_MX-zh_CN.es_MX 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_NI-zh_CN.es_NI 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_PA-zh_CN.es_PA 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_PE-zh_CN.es_PE 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_PR-zh_CN.es_PR 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_SV-zh_CN.es_SV 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_UY-zh_CN.es_UY 
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.es_VE-zh_CN.es_VE 
    https://object.pouta.csc.fi/OPUS-KDE4/v2/KDE4.es-zh_CN.es 
    https://object.pouta.csc.fi/OPUS-MultiUN/v1/MultiUN.es-zh.es 
    https://object.pouta.csc.fi/OPUS-News-Commentary/v14/News-Commentary.es-zh.es 
    https://object.pouta.csc.fi/OPUS-OpenOffice/v3/OpenOffice.es-zh_CN.es 
    https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/OpenSubtitles.es-zh_cn.es 
    https://object.pouta.csc.fi/OPUS-PHP/v1/PHP.es-zh.es 
    https://object.pouta.csc.fi/OPUS-QED/v2.0a/QED.es-zh.es 
    https://object.pouta.csc.fi/OPUS-Tanzil/v1/Tanzil.es-zh.es 
    https://object.pouta.csc.fi/OPUS-UN/v20090831/UN.es-zh.es 
    https://object.pouta.csc.fi/OPUS-UNPC/v1.0/UNPC.es-zh.es 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es-zh.es 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es-zh_CN.es 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_AR-zh_CN.es_AR 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_CL-zh_CN.es_CL 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_CO-zh_CN.es_CO 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_CR-zh_CN.es_CR 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_DO-zh_CN.es_DO 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_EC-zh_CN.es_EC 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_ES-zh_CN.es_ES 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_GT-zh_CN.es_GT 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_HN-zh_CN.es_HN 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_MX-zh_CN.es_MX 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_NI-zh_CN.es_NI 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_PA-zh_CN.es_PA 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_PE-zh_CN.es_PE 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_PR-zh_CN.es_PR 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_SV-zh_CN.es_SV 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_UY-zh_CN.es_UY 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.es_VE-zh_CN.es_VE 
    https://object.pouta.csc.fi/OPUS-bible-uedin/v1/bible-uedin.es-zh.es 
    https://object.pouta.csc.fi/OPUS-infopankki/v1/infopankki.es-zh.es
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.fr-zh_CN.fr 
    https://object.pouta.csc.fi/OPUS-infopankki/v1/infopankki.fr-zh.fr 
    https://object.pouta.csc.fi/OPUS-KDE4/v2/KDE4.fr-zh_CN.fr 
    https://object.pouta.csc.fi/OPUS-MultiUN/v1/MultiUN.fr-zh.fr 
    https://object.pouta.csc.fi/OPUS-News-Commentary/v14/News-Commentary.fr-zh.fr 
    https://object.pouta.csc.fi/OPUS-OpenOffice/v3/OpenOffice.fr-zh_CN.fr 
    https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/OpenSubtitles.fr-zh_cn.fr 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.fr_CA-zh_CN.fr_CA 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.fr_FR-zh_CN.fr_FR 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.frm-zh_CN.frm 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.fr-zh_CN.fr 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.fr-zh.fr 
    https://object.pouta.csc.fi/OPUS-UNPC/v1.0/UNPC.fr-zh.fr 
    https://object.pouta.csc.fi/OPUS-UN/v20090831/UN.fr-zh.fr
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.ko-zh_CN.ko 
    https://object.pouta.csc.fi/OPUS-KDE4/v2/KDE4.ko-zh_CN.ko 
    https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/OpenSubtitles.ko-zh_cn.ko 
    https://object.pouta.csc.fi/OPUS-PHP/v1/PHP.ko-zh.ko 
    https://object.pouta.csc.fi/OPUS-QED/v2.0a/QED.ko-zh.ko 
    https://object.pouta.csc.fi/OPUS-Tanzil/v1/Tanzil.ko-zh.ko 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.ko-zh.ko 
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.ko-zh_CN.ko 
    https://object.pouta.csc.fi/OPUS-bible-uedin/v1/bible-uedin.ko-zh.ko 
    https://object.pouta.csc.fi/OPUS-wikimedia/v20190628/wikimedia.ko-zh.ko
    https://object.pouta.csc.fi/OPUS-EUbookshop/v2/EUbookshop.pt-zh.pt  
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.pt-zh_CN.pt  
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.pt_BR-zh_CN.pt_BR  
    https://object.pouta.csc.fi/OPUS-GNOME/v1/GNOME.pt_PT-zh_CN.pt_PT  
    https://object.pouta.csc.fi/OPUS-KDE4/v2/KDE4.pt-zh_CN.pt  
    https://object.pouta.csc.fi/OPUS-KDE4/v2/KDE4.pt_BR-zh_CN.pt_BR  
    https://object.pouta.csc.fi/OPUS-News-Commentary/v14/News-Commentary.pt-zh.pt  
    https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/OpenSubtitles.pt-zh_cn.pt  
    https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/OpenSubtitles.pt_br-zh_cn.pt_br  
    https://object.pouta.csc.fi/OPUS-PHP/v1/PHP.pt_BR-zh.pt_BR  
    https://object.pouta.csc.fi/OPUS-QED/v2.0a/QED.pt-zh.pt  
    https://object.pouta.csc.fi/OPUS-Tanzil/v1/Tanzil.pt-zh.pt  
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.pt-zh.pt  
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.pt-zh_CN.pt  
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.pt_BR-zh.pt_BR  
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.pt_BR-zh_CN.pt_BR  
    https://object.pouta.csc.fi/OPUS-Ubuntu/v14.10/Ubuntu.pt_PT-zh_CN.pt_PT  
    https://object.pouta.csc.fi/OPUS-bible-uedin/v1/bible-uedin.pt-zh.pt"""


if __name__ == '__main__':
    main()