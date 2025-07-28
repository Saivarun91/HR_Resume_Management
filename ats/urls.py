from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.dashboard, name='dashboard'),
    path('download-candidates/', views.download_all_candidates, name='download_candidates'),
    path('candidate/<path:user_folder>/<path:candidate_folder>/', views.candidate_detail, name='candidate_detail'),
]