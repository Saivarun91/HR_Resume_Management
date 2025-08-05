from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import os, json, datetime, re, zipfile
from io import BytesIO
from .ResumeAnalytics import ResumeAnalytics
from django.template.loader import render_to_string
from xhtml2pdf import pisa

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'jpeg', 'jpg', 'png', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
analytics = ResumeAnalytics()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name.strip().replace(' ', '_'))

def get_user_folder(request, user_name):
    if not request.session.get('user_folder'):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        folder = f"{sanitize_name(user_name)}_{timestamp}"
        request.session['user_folder'] = folder
    return os.path.join('outputs', request.session['user_folder'])

def get_upload_folder(user_folder):
    upload_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    upload_folder = os.path.join(user_folder, f"upload_{upload_timestamp}")
    os.makedirs(upload_folder, exist_ok=True)
    return upload_folder

@csrf_exempt
def dashboard(request):
    if request.method == 'POST':
        user_name = request.POST.get('user_name', 'Anonymous')
        # Create a subfolder in outputs with the user_name
        main_user_folder = os.path.join('outputs', sanitize_name(user_name))
        os.makedirs(main_user_folder, exist_ok=True)
        # Use timestamped upload folder inside user_name folder
        upload_folder = get_upload_folder(main_user_folder)
        jd = request.FILES.get('jd')
        resumes = request.FILES.getlist('resumes')
        RELEVANCE_THRESHOLD = int(request.POST.get('relevance_threshold', 40))

        if not jd or not resumes:
            return render(request, "ats/dashboard.html", {
                "error": "Please upload a JD and one or more resumes.",
                "show_dashboard": False
            })

        fs = FileSystemStorage(location=UPLOAD_FOLDER)
        jd_path = fs.save(jd.name, jd)
        jd_full_path = os.path.join(UPLOAD_FOLDER, jd_path)

        # Save a copy of the JD in the upload_folder
        import shutil
        shutil.copy(jd_full_path, os.path.join(upload_folder, jd.name))

        candidates_data = []

        for resume_file in resumes:
            try:
                if not allowed_file(resume_file.name):
                    continue
                resume_path = fs.save(resume_file.name, resume_file)
                resume_full_path = os.path.join(UPLOAD_FOLDER, resume_path)
                # Save a copy of the resume in the upload_folder
                shutil.copy(resume_full_path, os.path.join(upload_folder, resume_file.name))
                ats_result = analytics.ATSanalytics(resume_full_path, output_folder=upload_folder)
                analytics_result = analytics.resumeanalytics(resume_full_path, jd_full_path, output_folder=upload_folder)
                ats_score = ats_result.get("ATS Score", {}).get("Total Score", 0)
                rrs_score = analytics_result.get("RESUME_RELEVANCE_SCORE", 0)
                rfs_score = analytics_result.get("ROLE_FIT_SCORE", 0)
                if rrs_score < RELEVANCE_THRESHOLD:
                    continue
                candidate_name = ats_result.get("Extracted Data", {}).get("Name", "Unknown")
                candidate_phone = ats_result.get("Extracted Data", {}).get("Contact Details", "unknown")
                folder_name = f"{sanitize_name(candidate_name)}_{sanitize_name(candidate_phone)}"
                candidate_folder = os.path.join(upload_folder, folder_name)
                os.makedirs(candidate_folder, exist_ok=True)
                # Copy resume and JD into candidate's folder
                shutil.copy(resume_full_path, os.path.join(candidate_folder, resume_file.name))
                shutil.copy(jd_full_path, os.path.join(candidate_folder, jd.name))
                # Save analysis results in candidate's folder
                report_data = {
                    "ATS": ats_result,
                    "Resume": analytics_result
                }
                with open(os.path.join(candidate_folder, "Candidate_Report.json"), "w", encoding="utf-8") as f:
                    json.dump(report_data, f, indent=4)
                candidates_data.append({
                    "filename": resume_file.name,
                    "folder": folder_name,
                    "ats_score": ats_score,
                    "rrs_score": rrs_score,
                    "rfs_score": rfs_score,
                    "components": ats_result.get("Extracted Data", {})
                })
            except Exception as e:
                print(f"Error processing {resume_file.name}: {e}")

        # Store the user folder path in session for download
        request.session["user_folder"] = os.path.relpath(upload_folder, 'outputs')
        return render(request, 'ats/dashboard.html', {
            'jd_filename': jd.name,
            'candidates_data': candidates_data,
            'show_dashboard': True,
            'user_folder': os.path.relpath(upload_folder, 'outputs')
        })
    return render(request, 'ats/dashboard.html', {'show_dashboard': False})

@csrf_exempt
def download_all_candidates(request):
    user_folder = request.session.get("user_folder")
    if not user_folder:
        return JsonResponse({"error": "No session folder found."}, status=400)
    base_folder = os.path.join("outputs", user_folder)
    candidate_folders = []
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path):
            report_path = os.path.join(item_path, "Candidate_Report.json")
            if os.path.exists(report_path):
                candidate_folders.append(item_path)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for folder in candidate_folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, base_folder)
                    zip_file.write(full_path, arcname)
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/zip')
    response['Content-Disposition'] = f'attachment; filename=qualified_candidates.zip'
    return response

def candidate_detail(request, user_folder, candidate_folder):
    # Path to the candidate's report
    report_path = os.path.join('outputs', user_folder, candidate_folder, 'Candidate_Report.json')
    if not os.path.exists(report_path):
        return HttpResponse('Analysis report not found.', status=404)
    with open(report_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)
    insights = analysis.get("Resume", {})
    if isinstance(insights, str):
        try:
            insights = json.loads(insights)
        except json.JSONDecodeError:
            insights = {}
    extracted = analysis.get("ATS", {}).get("Extracted Data", {})
    if isinstance(extracted, str):
        try:
            extracted = json.loads(extracted)
        except json.JSONDecodeError:
            extracted = {}
    return render(request, 'ats/candidate_detail.html', {
        'analysis': analysis,
        'insights': insights,
        'extracted': extracted,
        'user_folder': user_folder,
        'candidate_folder': candidate_folder
    })