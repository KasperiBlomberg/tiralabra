# Määrittelydokumentti
Kuulun tietojenkäsittelytieteen kandidaatti ohjelmaan.

## Aihe
Toteutan neuroverkon, joka tunnistaa kuvassa olevan vaatteen kategorian. Käytän aineistona [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) tietokantaa. Aineistossa on 60 000 kuvaa neuroverkon treenaamista varten ja 10 000 kuvaa testaamista varten. 
Kuvat ovat 28x28 harmaasävykuvia ja ne ovat luokiteltu kymmeneen eri kategoriaan. Projektin ydin on multilayer perceptron neuroverkko. 
Treenaan neuroverkkoa treeniaineistolla ja yritän saada testiaineiston luokitteluvirheen mahdollisimman pieneksi. Toteutan verkkoon tarvittavat algoritmit(vastavirta-algoritmi yms.) itse. Tavoitteena on saada aika- ja tilavaativuudet järkeviksi, tähän täytyy vielä perehtyä lisää.

## Ohjelmointikieli
Käytän projektin ohjelmointikielenä pythonia. Käytän matriisi- ja muihin laskutoimituksiin NumPy-kirjastoa. Vertaisarviointeja varten en hallitse muita kieliä.

## Projektin kieli
Kirjoitan projektin dokumentit suomeksi. Koodin ja kommentit kirjoitan englanniksi.

## Lähteet
Käytän ainakin seuraavia lähteitä

https://theneuralblog.com/forward-pass-backpropagation-example/

https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/

https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

https://tim.jyu.fi/view/143092#lis%C3%A4tietoa-aktivointifunktioista
