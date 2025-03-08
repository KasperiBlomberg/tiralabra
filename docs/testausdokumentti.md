## Yksikkötestit
Erilaisia apufunktioita on testattu `test_utils.py`tiedostossa. On testattu muun muassa datan lataamista ja esikäsittelyä. Tiedostossa on myös testattu neuroverkon käyttämät aktivointifunktiot. 
Funktioita on testattu suht yksinkertaisilla syötteillä, sillä funktiot itsessään ovat myös hyvin simppeleitä.

**Testikattavuusraportti**

![image](./coverage_screenshot.png)

Testien kattavuus on lähes 100, ainoastaan parametrien tallennus tiedosoon jää testaamatta.

## Integraatiotestaus
Itse neuroverkon toiminnan testaaminen on toteutettu integraatio testauksen avulla. En kokenut hyödylliseksi testata esimerkiksi vastavirtausalgoritmia erikseen, vaan tein suoraan päästä päätyyn testin sitä varten.
Testasin [tämän artikkelin](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/) mukaisia asioita kahden isomman testin avulla.
Kokeilin yhdellä test_train funktiolla, että neuroverkko ylisovittaa pieneen, tässä tapauksessa 10 samplea sisältävään, datasettiin. Samalla testasin että kaikki painot ja biasit päivittyvät jokaisen batchin jälkeen.
Testasin myös, että training loss pienenee jokaisen iteraation jälkeen ja että gradientit eivät ole ikinä 0. Syötteenä testi käyttää oikeaa dataa, eli syötteet ovat hyvin edustavia.
Lisäksi testasin toisella testillä, että samplejen järjestys ei vaikuta neuroverkon ennustuksiin.

Näiden testien pitäisi taata, että kaikki neuroverkon metodit toimivat oikein.
## Testien suorittaminen
Testit voi suorittaa virtuaaliympäristössä juurikansiossa komennolla `pytest`

## Verkon suorituskyky
Myös verkon hyvä suorituskyky kertoo jotain algoritmin toimivuudesta. Olen saavuttanut parhaillaan 88,9 % tarkkuuden testi datasetissä.
Zalandon repoon on linkattu samankaltainen MLP-neuroverkko, jonka kerrosten koot ovat 256, 128 ja 100. Kyseinen neuroverkko on saavuttanut tarkkuudeksi 88,3 %.
Mallini tehokkuus on siis samaa tasoa, josta voisi olettaa algoritmin toimivan oikein.
