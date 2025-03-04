import cupy as np  # CuPy-t használjuk NumPy helyett, GPU-n fut

ciklus_szam = 10000
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for asd in range(ciklus_szam):
        # szigmoid fugveny letrehozasa



    # szoval eloszor is en egy neuralis halot akarok es ehez en az xor logikai muveletet akarom hasznalni ami binaris tehat 0 vagy 1

    xor = np.array([[0, 0],
                    [0,1],
                    [1,0],
                    [1,1]])

    kivant_kimenet = np.array([[0],  # 0 XOR 0 → 0
                                [1],  # 0 XOR 1 → 1
                                [1],  # 1 XOR 0 → 1
                                [0]]) # 1 XOR 1 → 0


    # sulyok meg csinalasa

    suly1 = np.random.uniform(-1, 1, (2, 4))  # mit ahogy lahtaom a kepen minden neuron az elotte levovel es sajat magaval  tehat az elso   2 neuron a 4 el   
    suly2 = np.random.uniform(-1 , 1 ,(4,4,))  # 4 et a 4 el
    suly3 = np.random.uniform(-1 , 1 ,(4,1))   # 4 et az 1 el

    # biases letrehozasa   # ez segiti a neuralis halonak hogy jobban munkodjon gyakorlatilag  minden rejtett neuron kap egy biasest
    bias1 = np.random.uniform(-1, 1, (1, 4))  # Biasok az első rejtett réteghez
    bias2 = np.random.uniform(-1, 1, (1, 4))  # Biasok a második rejtett réteghez
    bias3 = np.random.uniform(-1, 1, (1, 1))  # Biasok a kimeneti réteghez
    # chatgpt azt mondta hogy : A biasok lehetővé teszik, hogy a neurális hálózatok jobb előrejelzéseket készítsenek és jobban alkalmazkodjanak a bemeneti adatokhoz. A bias egy "eltolás", amely segíti a 
    # neurális hálót abban, hogy ne ragadjon bele a tanulási folyamat során a nullás kimenetekbe vagy nehézkes bemenetekbe.



    #foward propagation(elorehaladas)  # elso rejtet reteg
    hidden_layer_input_1 = np.dot(xor, suly1)   + bias1 
    hidden_layer_output_1 = sigmoid(hidden_layer_input_1)   # ez a fugveny a sigmoid hoza letre nekunk az elso rejtet reteget  mert ugye ossze szoroza a sujt az inputal es ezt be rakja egy sigmoid fugvenybe ami letre hoz / elvegzi szamolast
    # es igy lehet hidden layert letre hozni

    # masodik reteg letrehozasa
    hidden_layer_input_2 = np.dot(hidden_layer_output_1 , suly2) + bias2
    hidden_layer_output_2 = sigmoid(hidden_layer_input_2)

    #kivant kimenet letrehozasa
    output_layer_input = np.dot(hidden_layer_output_2 , suly3)   + bias3
    output_layer_output = sigmoid(output_layer_input)

    # veszteseg kiszamitasa :   np.mean()  => ez ki szamitja az atlagot ,  es majd ezt a masodik hatanyra fogjuk emelni
    loss = np.mean((kivant_kimenet - output_layer_output) ** 2)     # na ez egyebkent azert kel mert a predikcionk(amit ki szamolt a kis neuronocska az nem eleg es ezt nekunk negyzetre kel emelni hogy meg kapjuk a jo erteket)



    # tanitas   (backpropagation)

    #1️⃣ lkimenet_hiba szamitas
    kimenet_hiba_szamitas = kivant_kimenet - output_layer_output

    # Kimeneti réteg gradiense (nem tudom mi a gecim ez mert nem vagyok matekos fasz)
    output_delta = kimenet_hiba_szamitas * output_layer_output * (1 - output_layer_output)
    """
    A output_error azt mutatja, hogy mennyire hibás a kimenet.
    A sigmoid deriváltja azt mutatja, hogy hogyan érzékeny a kimenet a bemenetre.
    E kettő szorzata adja meg, hogy mekkora módosítást kell végrehajtani a súlyokon, hogy a hiba csökkenjen.
    """


    #  Második rejtett réteg hibája és gradiens
    hidden_layer_2_error = output_delta.dot(suly3.T)  # Az előző réteg súlyait használjuk
    hidden_layer_2_delta = hidden_layer_2_error * hidden_layer_output_2 * (1 - hidden_layer_output_2)


    # elso rejtet reteg hibaja es gradiens
    hidden_layer_1_error = hidden_layer_2_delta.dot(suly2.T)
    hidden_layer_1_delta = hidden_layer_1_error * hidden_layer_output_1 * (1 - hidden_layer_output_1)



    # na most pedig a tanulasi sebeseget fogom be allitani
    learning_rate = 0.2


    # Sulyok frissítése
    suly3 += hidden_layer_output_2.T.dot(output_delta) * learning_rate
    suly2 += hidden_layer_output_1.T.dot(hidden_layer_2_delta) * learning_rate
    suly1 += xor.T.dot(hidden_layer_1_delta) * learning_rate

    # Biasok frissítése
    bias3 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    bias2 += np.sum(hidden_layer_2_delta, axis=0, keepdims=True) * learning_rate
    bias1 += np.sum(hidden_layer_1_delta, axis=0, keepdims=True) * learning_rate

 # Nyomtatás az iterációk alatt
    if ciklus_szam % 1000 == 0:  # Nyomtasd ki 1000 iterációnként a veszteséget
         print(f"Epoch {ciklus_szam}, Loss: {loss.get()}")