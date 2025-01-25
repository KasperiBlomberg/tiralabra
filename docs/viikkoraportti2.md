Tällä viikolla jatkoin algoritmin toimintaan ja sen taustalla olevaan matematiikkaan perehtymistä. Aloitin toteuttamaan algoritmia. 

Ohjelman toteutus on aloitettu. Datan pystyy nyt lataamaan ja esikäsittelemään. Esikäsittelyssä data kuvien pikselit normalisoidaan ja tunnisteet one-hot enkoodataan. Toteutin myös feedforward algoritmin ja backpropagation algoritmi on työn alla.

Tällä viikolla opin, millaisia käytännön haasteita neuraaliverkon toteuttamisessa voi esiintyä. 
Yksi haasteista liittyi feedforward-algoritmin käyttöön: yksittäisen koulutusnäytteen laskeminen on suoraviivaista, mutta tehokkuuden vuoksi näytteet kannattaa laskea samanaikaisesti matriisien avulla. Tämä kuitenkin lisää laskennan monimutkaisuutta.
Sama ongelma vaikuttaa myös backpropagation algoritmiin.

Feedforward- ja backpropagation-algoritmien kanssa on ollut jonkin verran haasteita, mutta olen pystynyt ratkaisemaan ne tähän mennessä. Uskon, että selviän myös jäljellä olevista ongelmista.

Seuraavaksi korjaan backpropagation algoritmin kuntoon ja alan tekemään testejä ohjelmalle. Kun saan toimivia testejä aikaan, algoritmin kehitys muuttuu helpommaksi. Koodin sisäinen laatu on myös todella heikkoa tällä hetkellä, joten laitan sen kuntoon. Lisään projektiin myös poetryn riippuvuuksien hallintaa varten.

Käytin tällä viikolla projektiin aikaa 10 tuntia.
