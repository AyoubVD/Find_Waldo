import cowsay

def helloWorld(x):
    d = cowsay.get_output_string('daemon', x)
    t = cowsay.get_output_string('trex', x)
    return d,t

x,t= helloWorld("yooo")
print("x")

