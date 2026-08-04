from django.shortcuts import render

# Create your views here.

def home(request):
    #return HttpResponse("<h1>Welcome to the Home page</h1>")
    return render(request, 'home.html', {'name': 'Greg Lim'})

def about(request):
    return render(request, 'home.html')