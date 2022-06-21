from django.urls import path

from mobile.views import (
    AppUserProfileAPI,
    AppUserRegistrationAPI,
    GetStatsAPI,
    ReportEWasteAPI,
    checkPredictorAPI,
)

urlpatterns = [
    path("register", AppUserRegistrationAPI.as_view(), name="register"),
    path("profile", AppUserProfileAPI.as_view(), name="app_user_profile"),
    path("stats", GetStatsAPI.as_view(), name="user_stats"),
    path("report", ReportEWasteAPI.as_view(), name="report_waste"),
    path("mltest", checkPredictorAPI.as_view(), name="test_ml"),
]
