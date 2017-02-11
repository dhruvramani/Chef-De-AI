#from text_generation import test
from model_bvc import test
from django.shortcuts import render

# Create your views here.
def genRec(request):
    if request.method == 'POST':
        sentence = request.POST.get('seed')
        gen = test(sentence)
        generated = []
        for i in gen :
            a = list(set(i.split("\n")))
            newList = []
            for j in a :
                new = unicode(str(j), errors='replace')
                newList.append(new)
            generated.append(newList)
        return render(request, 'Recipies/generated.html', {"genList" : generated})
    return render(request, 'Recipies/index.html', None)