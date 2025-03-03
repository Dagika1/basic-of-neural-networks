import numpy as np

# szigmoid fugveny letrehozasa
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

#foward propagation(elorehaladas)
hidden_layer_input_1 = np.dot(xor, suly1)    
hidden_layer_output_1 = sigmoid(hidden_layer_input_1)   # ez a fugveny a sigmoid hoza letre nekunk az elso rejtet reteget  mert ugye ossze szoroza a sujt az inputal es ezt be rakja egy sigmoid fugvenybe ami letre hoz / elvegzi szamolast
# es igy lehet hidden layert letre hozni

# masodik reteg letrehozasa
hidden_layer_input_2 = np.dot(hidden_layer_output_1 , suly2)
hidden_layer_output_2 = sigmoid(hidden_layer_input_2)   
#https://chatgpt.com/c/67c4cc64-80d8-8009-aa03-b2d4003d1a6c
