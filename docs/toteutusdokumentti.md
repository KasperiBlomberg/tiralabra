# Toteutusdokumentti
## Ohjelman yleisrakenne
Ohjelma tunnistaa vaatteita ja asusteita kuvista eri kategorioihin. Ohjelman ydin on on multilayer perceprton-neuroverkko. Ohjelma on toteutettu Pythonilla ja matriisilaskuihin on käytetty NumPy-kirjastoa.
Neuroverkko koostuu 784 neuronin sisääntulokerroksesta, kahdesta piilotetusta kerroksesta (256 ja 128) neuronia sekä ulostulokerroksesta, jossa on 10 neuronia eli neuroni jokaista luokkaa kohti.
Verkko käyttää aktivointifunktiona ReLu:a ja ulostulokerroksessa softmaxia.

Verkko käyttää koulutuksessa L2-säännöllistämistä parametrejen päivittämiseen, jotta painot eivät kasva liikaa. Koulutuksessa käytetään myös mini-batchejä ja batchejen järjestys sekoitetaan jokaisessa iteraatiossa.

Parhaillaan ohjelmalla on saavutettu n. 94 % tarkuus koulutusdatassa ja n. 89 % tarkkuus testidatassa. Zalandon mukaan ihmiset tunnistavat kuvista oikein 83,5 % ja suht vastaavanlainen MLP (kerrosten koot 256-128-100) on tunnistanut 88 % kuvista, joten tulokset ovat varsin hyviä. 

Ohjelma koostuu seuraavista komponenteista:
- `neural_network.py` Pääkomponentti, joka sisältää neuroverkon rakenteen ja metodit neuroverkon toiminnalle.
- `scripts` Sisältää skriptit neuroverkon kouluttamista ja evaluaatiota varten.
- `activation_functions.py` Sisältää neuroverkon tarvitsemat aktivointifunktiot. 
- `data_loader.py` ja `mnist_reader.py` mnistreader-tiedosto on suoraan kopioitu Zalandon [reposta](https://github.com/zalandoresearch/fashion-mnist) ja sitä on käytetty kuvien lataamiseen tiedostosta. Data_loader komponentissa on funktioita datan esikäsittelyyn esimerkiksi normalisointia ja tunnisteiden enkoodausta varten.
- `main.py` Sisältää käyttöliittymän.

## Saavutetut aika- ja tilavaativuudet
Todo

## Työn mahdolliset puutteet ja parannusehdotukset
Käyttöliittymää voisi vielä parannella.
Hyperparametrejä voisi optimoida ja neuroverkon syvyyttä tai leveyttä kasvattaa. Kuvia voisi myös esikäsitellä esimerkiksi kääntämällä kuvia. Näillä voitaisiin päästä parempiin tarkkuuksiin, mutta tämänhetkisellä toteuksella on kuitenkin päästy varsin hyviin tuloksiin.

## Laajojen kielimallien käyttö
Projektissa on käytetty apuna ChatGPT:tä:
- Tiedonhakuun
- Virheilmoitusten debuggaamiseen
- Joidenkin konseptien selittämiseen
- Dokumenttien oikeinkirjoituksen ja selkeyden parantamiseen

Yhtään koodia ei ole tuotettu kielimallien avulla.

## Viitteet
https://www.youtube.com/watch?v=sIX_9n-1UbM

https://theneuralblog.com/forward-pass-backpropagation-example/

https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/

https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

https://tim.jyu.fi/view/143092#lis%C3%A4tietoa-aktivointifunktioista
