mkdir interface/model
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1w7-SzstV0PhjnuOll6VSMAt-tncgMMAu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1w7-SzstV0PhjnuOll6VSMAt-tncgMMAu" -O interface/model/model.ckpt-first.data-00000-of-00001 && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BSELc2fG1Ka8S8lmKZj2HOGqEHapquvj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BSELc2fG1Ka8S8lmKZj2HOGqEHapquvj" -O interface/model/model.ckpt-first.meta && rm -rf /tmp/cookies.txt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=17KNUgEREa4IOwhFNYeN50bD30qDzYaQf' -O interface/model/model.ckpt-first.index
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GupyDldJOyx1nch1hYdwUizcmDIj0B7a' -O interface/model/words_cnt.txt