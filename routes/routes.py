from flask import request, render_template, jsonify, session, redirect, url_for
from werkzeug.security import check_password_hash, generate_password_hash
import logging
from .models import db, User
from .rag_core import get_rag_response, initialize_rag_system
import time
from datetime import datetime
from uuid import uuid4
import json

# ============ Topic Extraction Utilities (phrase-based) ============
import re as _re
import collections as _collections

def _normalize_text(text):
    if not text:
        return ""
    return _re.sub(r"\s+", " ", text.lower()).strip()

# Expanded stopwords including 'cara', interrogatives, and common fillers
STOPWORDS = set([
    'dan','atau','yang','di','ke','dari','untuk','pada','dengan','apa','bagaimana','adalah','saya','ini','itu',
    'the','of','in','to','a','is','are','it','as','by','an','be','on','for','was','were','has','have','had','will','can',
    'petani','chatbot','tolong','mohon','apakah','seperti','agar','yang','jika','bila','dapat','bisa','cara'
])

# Some phrase starters to avoid as topics
_AVOID_START = {'cara', 'bagaimana', 'apa', 'tolong', 'mohon'}

_TOKEN_RE = _re.compile(r"\b\w+\b", flags=_re.UNICODE)

def tokenize_words(text):
    return _TOKEN_RE.findall(_normalize_text(text))

def extract_phrases(text, ngram_sizes=(2,3)):
    """Extract candidate phrases (bigrams/trigrams) from text, avoiding stopword-only phrases
    and removing phrases starting/ending with stopwords.
    """
    tokens = tokenize_words(text)
    phrases = []
    for n in ngram_sizes:
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            window = tokens[i:i+n]
            if window[0] in STOPWORDS or window[-1] in STOPWORDS or window[0] in _AVOID_START:
                continue
            # require at least one non-stopword token of length > 2
            if not any((t not in STOPWORDS and len(t) > 2) for t in window):
                continue
            phrase = ' '.join(window)
            phrases.append(phrase)
    return phrases

def extract_best_phrase(text, global_phrase_counts=None):
    """Pick the best phrase from text based on global counts; fallback to the first meaningful word.
    """
    candidates = extract_phrases(text)
    if candidates:
        if global_phrase_counts:
            # choose highest frequency phrase present in text
            candidates_sorted = sorted(candidates, key=lambda p: global_phrase_counts.get(p, 0), reverse=True)
            if candidates_sorted:
                return candidates_sorted[0]
        return candidates[0]
    # fallback to first meaningful word
    for w in tokenize_words(text):
        if w not in STOPWORDS and len(w) > 2 and w not in _AVOID_START:
            return w
    return '-'



def from_json_filter(value):
    import json
    if not value:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return []
    return value

def register_routes(app):

    @app.route('/delete_session', methods=['POST'])
    def delete_session():
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        data = request.get_json()
        session_id = data.get('session_id')
        user_id = session.get('user_id')
        if not session_id:
            return jsonify({'error': 'Session ID tidak valid'}), 400
        from .models import ChatHistory
        print(f"[DEBUG] Hapus session_id={session_id}, user_id={user_id}")
        deleted = ChatHistory.query.filter_by(session_id=session_id).delete()
        db.session.commit()
        print(f"[DEBUG] Jumlah chat dihapus: {deleted}")
        return jsonify({'success': True, 'deleted': deleted})
    # /logout route should only be registered once. Remove duplicate if exists elsewhere.
    app.jinja_env.filters['from_json'] = lambda value: json.loads(value) if value else []

    @app.route('/start_session', methods=['POST'])
    def start_session():
        session_id = str(uuid4())
        return jsonify({'session_id': session_id})

    @app.route('/')
    @app.route('/index')
    def index():
        if 'user_id' not in session:
            return redirect(url_for('login'))
        from .models import ChatHistory
        user_id = session.get('user_id')
        # Ambil semua session milik user, urut terbaru
        all_sessions = db.session.query(ChatHistory.session_id, db.func.min(ChatHistory.created_at))\
            .filter(ChatHistory.user_id == user_id)\
            .group_by(ChatHistory.session_id)\
            .order_by(db.func.max(ChatHistory.created_at).desc())\
            .all()
        sessions = []
        for sid, _ in all_sessions:
            first_chat = ChatHistory.query.filter_by(session_id=sid, user_id=user_id).order_by(ChatHistory.created_at).first()
            if first_chat:
                title = first_chat.question[:40] if first_chat.question else '(Chat Kosong)'
                sessions.append({'session_id': sid, 'title': title})
        # Ambil session aktif dari query param
        session_id = request.args.get('session')
        chats = []
        if session_id:
            chats = ChatHistory.query.filter_by(session_id=session_id, user_id=user_id).order_by(ChatHistory.created_at).all()
        elif sessions:
            # Default ke session terbaru jika tidak ada param
            session_id = sessions[0]['session_id']
            chats = ChatHistory.query.filter_by(session_id=session_id, user_id=user_id).order_by(ChatHistory.created_at).all()
        return render_template('index.html', sessions=sessions, active_session=session_id, active_chats=chats)

    @app.route('/dashboard_admin')
    def dashboard_admin():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return redirect(url_for('login'))
        from .models import ChatHistory, User, Region, Province, Regency
        from sqlalchemy import func
        from datetime import datetime, timedelta
        import collections, re

        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)

        # Province/Regency filter
        selected_region = request.args.get('region', '').strip()  # regency name
        selected_province_id = request.args.get('province_id', '').strip()
        province_obj = None
        if selected_region and not selected_province_id:
            # infer province from regency name for UI selection
            reg_obj = Regency.query.filter(Regency.name == selected_region).first()
            if reg_obj:
                selected_province_id = str(reg_obj.province_id)
        if selected_province_id:
            province_obj = Province.query.get(int(selected_province_id))

        # Ensure regencies exist for selected province (auto-seed on demand)
        if selected_province_id:
            try:
                pid_int = int(selected_province_id)
                if Regency.query.filter_by(province_id=pid_int).count() == 0:
                    import requests
                    base = 'https://emsifa.github.io/api-wilayah-indonesia/api'
                    prov = Province.query.get(pid_int)
                    if prov:
                        # Map province name to API id
                        resp = requests.get(f'{base}/provinces.json', timeout=20)
                        resp.raise_for_status()
                        provinces_json = resp.json() or []
                        api_pid = None
                        for pj in provinces_json:
                            if pj.get('name') == prov.name:
                                api_pid = pj.get('id')
                                break
                        if api_pid:
                            rresp = requests.get(f'{base}/regencies/{api_pid}.json', timeout=20)
                            if rresp.status_code == 200:
                                rjson = rresp.json() or []
                                for rj in rjson:
                                    rname = rj.get('name')
                                    if not rname:
                                        continue
                                    exists = Regency.query.filter_by(name=rname, province_id=pid_int).first()
                                    if not exists:
                                        db.session.add(Regency(name=rname, province_id=pid_int))
                                db.session.commit()
            except Exception:
                db.session.rollback()
                # silently ignore, UI will show empty list

        # apply filters
        if selected_region:
            chats = ChatHistory.query.join(User, ChatHistory.user_id == User.id) \
                .filter(ChatHistory.created_at >= week_ago, User.region == selected_region).all()
        elif selected_province_id:
            chats = ChatHistory.query.join(User, ChatHistory.user_id == User.id) \
                .join(Regency, Regency.name == User.region) \
                .filter(ChatHistory.created_at >= week_ago, Regency.province_id == int(selected_province_id)).all()
        else:
            chats = ChatHistory.query.filter(ChatHistory.created_at >= week_ago).all()

        jumlah_pertanyaan = len(chats)

        # Build phrase counts over last week
        phrase_counter = _collections.Counter()
        for c in chats:
            phrase_counter.update(extract_phrases(c.question))
        # fallback: if no phrases at all, use keywords excluding stopwords
        if not phrase_counter:
            all_questions = ' '.join([c.question or '' for c in chats]).lower()
            keywords = re.findall(r'\b\w+\b', all_questions)
            keywords = [w for w in keywords if w not in STOPWORDS and len(w) > 2]
            phrase_counter = _collections.Counter(keywords)
        top_topics = phrase_counter.most_common(5)

        # All region & topic for filter (union Region master + user regions)
        user_regions = [u.region for u in User.query.filter(User.region != None).with_entities(User.region).all()]
        master_regions = [r.name for r in Region.query.order_by(Region.name.asc()).all()]
        all_regions = sorted(set([r for r in user_regions if r] + master_regions))
        all_topics = [t for t, _ in phrase_counter.most_common(15)]

        # All chats for table (join user)
        user_map = {u.id: u for u in User.query.all()}

        def extract_topik(q):
            return extract_best_phrase(q, global_phrase_counts=phrase_counter)

        all_chats = [
            {
                'question': c.question,
                'answer': c.answer,
                'created_at': c.created_at,
                'user_name': user_map[c.user_id].name if c.user_id in user_map else '-',
                'region': user_map[c.user_id].region if c.user_id in user_map and user_map[c.user_id].region else '-',
                'topik': extract_topik(c.question)
            }
            for c in chats
        ]

        # Insight perilaku pengguna
        jam_counter = _collections.Counter([c.created_at.hour for c in chats if c.created_at])
        jam_aktif_tertinggi = f"{jam_counter.most_common(1)[0][0]}:00" if jam_counter else '-'
        user_counter = _collections.Counter([c.user_id for c in chats if c.user_id])
        rata2_pertanyaan_per_user = f"{(jumlah_pertanyaan/len(user_counter)):.2f}" if user_counter else '-'

        # Wilayah aktif (tanpa koordinat atau username)
        regions_set = set()
        for c in chats:
            if c.user_id in user_map and user_map[c.user_id].region:
                regions_set.add(user_map[c.user_id].region)
        regions_in_chats = sorted(regions_set)
        wilayah_aktif = ', '.join(regions_in_chats) if regions_in_chats else '-'

        # Early Warning System (setelah data tersedia)
        ews = []
        ews_map = {}
        for c in chats:
            user = user_map.get(c.user_id)
            region = user.region if user and user.region else '-'
            topik = extract_topik(c.question)
            key = (region, topik)
            if key not in ews_map:
                ews_map[key] = {'users': set(), 'count': 0}
            if c.user_id:
                ews_map[key]['users'].add(c.user_id)
            ews_map[key]['count'] += 1
        for (region, topik), v in ews_map.items():
            if region != '-' and topik != '-' and len(v['users']) >= 3 and v['count'] >= 5:
                ews.append({
                    'region': region,
                    'topik': topik,
                    'user_count': len(v['users']),
                    'chat_count': v['count']
                })

        return render_template('dashboard_admin.html',
            user=session.get('user_name'),
            jumlah_pertanyaan=jumlah_pertanyaan,
            top_topics=top_topics,
            wilayah_aktif=wilayah_aktif,
            all_regions=all_regions,
            all_topics=all_topics,
            all_chats=all_chats,
            jam_aktif_tertinggi=jam_aktif_tertinggi,
            rata2_pertanyaan_per_user=rata2_pertanyaan_per_user,
            ews=ews,
            selected_region=selected_region,
            selected_province_id=selected_province_id or ''
        )



    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if 'user_id' in session:
            if session.get('user_role') == 'admin':
                return redirect(url_for('dashboard_admin'))
            return redirect(url_for('index'))
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password']
            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password, password):
                session['user_id'] = user.id
                session['user_name'] = user.name
                session['is_admin'] = user.role == 'admin'
                session['user_role'] = user.role
                session['admin_region'] = user.region
                if user.role == 'admin':
                    return redirect(url_for('dashboard_admin'))
                else:
                    return redirect(url_for('index'))
            else:
                return render_template('login.html', error="Email atau password salah.")
        return render_template('login.html')

    @app.route('/daftar', methods=['GET', 'POST'])
    def daftar_petani():
        if 'user_id' in session:
            return redirect(url_for('index'))
        if request.method == 'POST':
            data = request.form
            if User.query.filter_by(email=data['email']).first():
                return render_template('daftar.html', error="Email sudah terdaftar.")
            user = User(
                name=data['name'],
                dob=datetime.strptime(data['dob'], '%Y-%m-%d'),
                gender=data['gender'],
                email=data['email'],
                phone=data['phone'],
                password=generate_password_hash(data['password']),
                role='petani'
            )
            db.session.add(user)
            db.session.commit()
            return redirect(url_for('login'))
        return render_template('daftar.html')

    @app.route('/logout')
    def logout():
        session.clear()
        return redirect(url_for('login'))

    
    
    @app.route('/chat', methods=['POST'])
    def chat():
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        """Optimized chat endpoint with better error handling and performance"""
        start_time = time.time()
        try:
            # Validate request
            if not request.is_json:
                return jsonify({'error': 'Request must be JSON'}), 400
            
            data = request.get_json()
            if not data or 'message' not in data:
                return jsonify({'error': 'Message is required'}), 400
            
            user_question = data['message'].strip()
            if not user_question:
                return jsonify({'error': 'Message cannot be empty'}), 400
            
            # Limit message length
            if len(user_question) > 1000:
                return jsonify({'error': 'Message too long. Please keep it under 1000 characters.'}), 400
            
            history = data.get('history', [])
            
            # Limit history to prevent context overflow
            if len(history) > 10:
                history = history[-10:]
            
            
            # Get response from optimized RAG system
            response_text, sources = get_rag_response(user_question, history)
            # Jika jawaban adalah pesan default/tidak relevan, kosongkan sources
            if response_text.strip().lower().startswith("mohon maaf") or response_text.strip().lower().startswith("maaf, pertanyaan anda"):
                sources = []

            # Simpan ke riwayat chat jika user login
            from .models import ChatHistory
            session_id = data.get('session_id')
            user_id = session.get('user_id')
            if not session_id:
                # Buat session baru jika tidak ada
                session_id = str(uuid4())
            chat_entry = ChatHistory(
                user_id=user_id,
                question=user_question,
                answer=response_text,
                sources=json.dumps(sources) if sources else None,
                session_id=session_id
            )
            db.session.add(chat_entry)
            db.session.commit()

            processing_time = time.time() - start_time

            return jsonify({
                'reply': response_text,
                'sources': sources,
                'processing_time': round(processing_time, 2),
                'session_id': session_id
            })
        except Exception as e:
            error_id = int(time.time())
            logging.exception(f"[ERROR][chat] ID: {error_id}")
            # Jika mode debug, tampilkan error detail di response
            import os
            debug = os.environ.get('FLASK_DEBUG', '0') == '1'
            return jsonify({
                'reply': f"Maaf, terjadi kesalahan server (ID: {error_id}). Silakan coba lagi atau hubungi administrator jika masalah berlanjut.",
                'error_id': error_id,
                'error_detail': str(e) if debug else None
            }), 500

    @app.route('/chat/health', methods=['GET'])
    def chat_health():
        """Health check endpoint for chat system"""
        try:
            # Quick test of RAG system
            test_response, _ = get_rag_response("test", [])
            
            return jsonify({
                'status': 'healthy',
                'rag_system': 'operational' if test_response else 'unavailable',
                'timestamp': datetime.utcnow().isoformat()
            })
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 500

    @app.route('/chat/init', methods=['POST'])
    def initialize_chat():
        """Endpoint to manually initialize RAG system"""
        try:
            initialize_rag_system()
            return jsonify({
                'status': 'initializing',
                'message': 'RAG system initialization started'
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @app.route('/api/admin/provinces', methods=['GET'])
    def api_admin_provinces():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 401
        from .models import Province, Regency
        provs = Province.query.order_by(Province.name.asc()).all()
        if not provs:
            # Auto-seed once from public API if empty
            try:
                import requests
                base = 'https://emsifa.github.io/api-wilayah-indonesia/api'
                # fetch provinces
                resp = requests.get(f'{base}/provinces.json', timeout=20)
                resp.raise_for_status()
                provinces_json = resp.json() or []
                for pj in provinces_json:
                    name = pj.get('name')
                    if not name:
                        continue
                    if not Province.query.filter_by(name=name).first():
                        db.session.add(Province(name=name))
                db.session.commit()
                # fetch regencies per province
                provs = Province.query.order_by(Province.name.asc()).all()
                for p in provs:
                    # find id from API by matching name
                    api_pid = None
                    for pj in provinces_json:
                        if pj.get('name') == p.name:
                            api_pid = pj.get('id')
                            break
                    if not api_pid:
                        continue
                    rresp = requests.get(f'{base}/regencies/{api_pid}.json', timeout=20)
                    if rresp.status_code != 200:
                        continue
                    rjson = rresp.json() or []
                    for rj in rjson:
                        rname = rj.get('name')
                        if not rname:
                            continue
                        exists = Regency.query.filter_by(name=rname, province_id=p.id).first()
                        if not exists:
                            db.session.add(Regency(name=rname, province_id=p.id))
                db.session.commit()
                provs = Province.query.order_by(Province.name.asc()).all()
            except Exception:
                db.session.rollback()
                # fall back to empty if seeding fails
        return jsonify({'provinces': [{'id': p.id, 'name': p.name} for p in provs]})

    @app.route('/api/admin/seed-indonesia-regions', methods=['POST'])
    def api_admin_seed_indonesia_regions():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 401
        from .models import Province, Regency
        try:
            import requests
            base = 'https://emsifa.github.io/api-wilayah-indonesia/api'
            resp = requests.get(f'{base}/provinces.json', timeout=30)
            resp.raise_for_status()
            provinces_json = resp.json() or []
            name_to_id = {}
            for pj in provinces_json:
                name = pj.get('name')
                if not name:
                    continue
                p = Province.query.filter_by(name=name).first()
                if not p:
                    p = Province(name=name)
                    db.session.add(p)
                    db.session.flush()
                name_to_id[name] = (pj.get('id'), p.id)
            db.session.commit()
            count_regs = 0
            for name, (api_id, db_id) in name_to_id.items():
                try:
                    rresp = requests.get(f'{base}/regencies/{api_id}.json', timeout=30)
                    rresp.raise_for_status()
                    rjson = rresp.json() or []
                    for rj in rjson:
                        rname = rj.get('name')
                        if not rname:
                            continue
                        exists = Regency.query.filter_by(name=rname, province_id=db_id).first()
                        if not exists:
                            db.session.add(Regency(name=rname, province_id=db_id))
                            count_regs += 1
                    db.session.commit()
                except Exception:
                    db.session.rollback()
                    continue
            return jsonify({'ok': True, 'provinces': len(name_to_id), 'regencies_added': count_regs})
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': 'Seeding gagal'}), 500

    @app.route('/api/admin/regencies', methods=['GET'])
    def api_admin_regencies():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 401
        from .models import Regency, Province
        try:
            pid = int(request.args.get('province_id', '0'))
        except Exception:
            pid = 0
        if not pid:
            return jsonify({'regencies': []})
        regs = Regency.query.filter_by(province_id=pid).order_by(Regency.name.asc()).all()
        # Auto-seed regencies for this province if empty
        if not regs:
            try:
                import requests
                base = 'https://emsifa.github.io/api-wilayah-indonesia/api'
                prov = Province.query.get(pid)
                if prov:
                    resp = requests.get(f'{base}/provinces.json', timeout=20)
                    resp.raise_for_status()
                    provinces_json = resp.json() or []
                    api_pid = None
                    for pj in provinces_json:
                        if pj.get('name') == prov.name:
                            api_pid = pj.get('id')
                            break
                    if api_pid:
                        rresp = requests.get(f'{base}/regencies/{api_pid}.json', timeout=20)
                        if rresp.status_code == 200:
                            rjson = rresp.json() or []
                            for rj in rjson:
                                rname = rj.get('name')
                                if not rname:
                                    continue
                                exists = Regency.query.filter_by(name=rname, province_id=pid).first()
                                if not exists:
                                    db.session.add(Regency(name=rname, province_id=pid))
                            db.session.commit()
                            regs = Regency.query.filter_by(province_id=pid).order_by(Regency.name.asc()).all()
            except Exception:
                db.session.rollback()
                regs = []
        return jsonify({'regencies': [{'id': r.id, 'name': r.name} for r in regs]})

    # --- Regions master management ---
    @app.route('/api/admin/regions', methods=['GET', 'POST'])
    def api_admin_regions():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 401
        from .models import Region, User
        if request.method == 'GET':
            master = [r.name for r in Region.query.order_by(Region.name.asc()).all()]
            user_regions = [r[0] for r in User.query.filter(User.region != None).with_entities(User.region).distinct().all()]
            all_regions = sorted(set([*(master or []), *([ur for ur in user_regions if ur] or [])]))
            return jsonify({'regions': all_regions, 'master': master})
        # POST: add new region
        try:
            data = request.get_json() or {}
            name = (data.get('name') or '').strip()
            if not name:
                return jsonify({'error': 'Nama wilayah wajib diisi'}), 400
            exists = Region.query.filter(Region.name.ilike(name)).first()
            if exists:
                return jsonify({'ok': True, 'message': 'Wilayah sudah ada'}), 200
            reg = Region(name=name)
            db.session.add(reg)
            db.session.commit()
            return jsonify({'ok': True, 'name': name}), 201
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': 'Gagal menambah wilayah'}), 500

    @app.route('/api/admin/trends', methods=['GET'])
    def api_admin_trends():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 401
        from .models import ChatHistory, User, Regency, Province
        from datetime import timedelta
        import collections, re

        period = request.args.get('period', 'week').lower()  # day|week|month
        if period not in {'day', 'week', 'month'}:
            period = 'week'
        try:
            days = int(request.args.get('days', '90'))
        except Exception:
            days = 90
        try:
            top_k = int(request.args.get('top_k', '5'))
        except Exception:
            top_k = 5

        region_filter = request.args.get('region', '').strip()
        province_id = request.args.get('province_id', '').strip()

        # Ensure regencies exist for requested province
        if province_id:
            try:
                pid_int = int(province_id)
                if Regency.query.filter_by(province_id=pid_int).count() == 0:
                    import requests
                    base = 'https://emsifa.github.io/api-wilayah-indonesia/api'
                    prov = Province.query.get(pid_int)
                    if prov:
                        resp = requests.get(f'{base}/provinces.json', timeout=20)
                        resp.raise_for_status()
                        provinces_json = resp.json() or []
                        api_pid = None
                        for pj in provinces_json:
                            if pj.get('name') == prov.name:
                                api_pid = pj.get('id')
                                break
                        if api_pid:
                            rresp = requests.get(f'{base}/regencies/{api_pid}.json', timeout=20)
                            if rresp.status_code == 200:
                                rjson = rresp.json() or []
                                for rj in rjson:
                                    rname = rj.get('name')
                                    if not rname:
                                        continue
                                    exists = Regency.query.filter_by(name=rname, province_id=pid_int).first()
                                    if not exists:
                                        db.session.add(Regency(name=rname, province_id=pid_int))
                                db.session.commit()
            except Exception:
                db.session.rollback()
                # ignore

        now = datetime.utcnow()
        start_time = now - timedelta(days=days)
        if region_filter:
            chats = ChatHistory.query.join(User, ChatHistory.user_id == User.id) \
                .filter(ChatHistory.created_at >= start_time, User.region == region_filter).all()
        elif province_id:
            # Get all regency names for the selected province
            try:
                pid_int = int(province_id)
                regency_names = [r.name for r in Regency.query.filter_by(province_id=pid_int).all()]
                
                # Debug logging
                print(f"DEBUG: province_id={province_id}, regency_names count={len(regency_names)}")
                if regency_names:
                    print(f"DEBUG: First 5 regency names: {regency_names[:5]}")
                
                if regency_names:
                    # Strategy 1: Exact match with regency names
                    chats = ChatHistory.query.join(User, ChatHistory.user_id == User.id) \
                        .filter(ChatHistory.created_at >= start_time, User.region.in_(regency_names)).all()
                    print(f"DEBUG: Found {len(chats)} chats with exact regency match")
                    
                    # Strategy 2: If no exact match, try fuzzy matching with province name
                    if len(chats) == 0:
                        province = Province.query.get(pid_int)
                        if province:
                            province_name = province.name.lower()
                            # Find users whose region contains province name (fuzzy match)
                            fuzzy_chats = ChatHistory.query.join(User, ChatHistory.user_id == User.id) \
                                .filter(ChatHistory.created_at >= start_time) \
                                .filter(db.func.lower(User.region).contains(province_name)).all()
                            chats = fuzzy_chats
                            print(f"DEBUG: Fuzzy match found {len(chats)} chats for province '{province_name}'")
                else:
                    # If no regencies found, try fallback with province name directly
                    province = Province.query.get(pid_int)
                    if province:
                        province_name = province.name.lower()
                        # Try to find users with region containing province name
                        chats = ChatHistory.query.join(User, ChatHistory.user_id == User.id) \
                            .filter(ChatHistory.created_at >= start_time) \
                            .filter(db.func.lower(User.region).contains(province_name)).all()
                        print(f"DEBUG: Fallback strategy found {len(chats)} chats for province '{province_name}'")
                    else:
                        chats = []
                        print(f"DEBUG: No regencies and no province found for ID {province_id}")
            except ValueError:
                # Invalid province_id, return empty result
                chats = []
                print(f"DEBUG: Invalid province_id: {province_id}")
            except Exception as e:
                # Handle any other database errors
                chats = []
                print(f"DEBUG: Database error for province_id {province_id}: {str(e)}")
        else:
            chats = ChatHistory.query.filter(ChatHistory.created_at >= start_time).all()

        # Compute phrase counts across chats and choose a best phrase per chat
        phrase_counts = _collections.Counter()
        chat_best_phrase = {}
        for c in chats:
            phs = extract_phrases(c.question)
            phrase_counts.update(phs)
        # Fallback to keywords when phrase extraction yields nothing
        if not phrase_counts:
            all_questions = ' '.join([(c.question or '') for c in chats]).lower()
            keywords = re.findall(r'\b\w+\b', all_questions)
            keywords = [w for w in keywords if w not in STOPWORDS and len(w) > 2]
            phrase_counts = _collections.Counter(keywords)
        for c in chats:
            chat_best_phrase[c.id] = extract_best_phrase(c.question, global_phrase_counts=phrase_counts)

        def bucket_start(dt):
            if period == 'day':
                return datetime(dt.year, dt.month, dt.day)
            if period == 'week':
                # set to Monday 00:00
                monday = dt - timedelta(days=dt.weekday())
                return datetime(monday.year, monday.month, monday.day)
            # month
            return datetime(dt.year, dt.month, 1)

        # Aggregate counts
        totals_by_topic = collections.Counter()
        counts = {}
        for c in chats:
            if not c.created_at:
                continue
            bucket = bucket_start(c.created_at)
            topic = chat_best_phrase.get(c.id, '-')
            if topic == '-':
                continue
            totals_by_topic[topic] += 1
            counts.setdefault(bucket, {}).setdefault(topic, 0)
            counts[bucket][topic] += 1

        top_topics = [t for t, _ in totals_by_topic.most_common(top_k)]
        # Build sorted bucket labels
        labels = sorted(counts.keys())
        # Ensure continuous buckets and also build when no data
        def next_bucket(d):
            if period == 'day':
                return d + timedelta(days=1)
            if period == 'week':
                return d + timedelta(days=7)
            # month increment
            year = d.year + (1 if d.month == 12 else 0)
            month = 1 if d.month == 12 else d.month + 1
            return datetime(year, month, 1)
        if labels:
            filled = []
            cursor = labels[0]
            last = labels[-1]
            while cursor <= last:
                filled.append(cursor)
                cursor = next_bucket(cursor)
            labels = filled
        else:
            # No data: still generate timeline buckets for the selected range
            start_bucket = bucket_start(start_time)
            end_bucket = bucket_start(now)
            cursor = start_bucket
            labels = []
            max_steps = 1000
            while cursor <= end_bucket and max_steps > 0:
                labels.append(cursor)
                cursor = next_bucket(cursor)
                max_steps -= 1

        # Build matrix
        matrix = []
        for topic in top_topics:
            row = []
            for b in labels:
                row.append(counts.get(b, {}).get(topic, 0))
            matrix.append(row)

        # Serialize labels as ISO date strings
        label_strings = [b.strftime('%Y-%m-%d') for b in labels]
        return jsonify({
            'labels': label_strings,
            'topics': top_topics,
            'matrix': matrix
        })

    @app.route('/api/admin/topic-distribution', methods=['GET'])
    def api_admin_topic_distribution():
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return jsonify({'error': 'Unauthorized'}), 401
        from .models import ChatHistory, User
        from datetime import timedelta
        import collections, re

        topic = request.args.get('topic', '').strip().lower()
        try:
            days = int(request.args.get('days', '30'))
        except Exception:
            days = 30
        region_filter = request.args.get('region', '').strip()
        province_id = request.args.get('province_id', '').strip()

        # Ensure regencies exist for requested province
        if province_id:
            try:
                from .models import Province, Regency
                pid_int = int(province_id)
                if Regency.query.filter_by(province_id=pid_int).count() == 0:
                    import requests
                    base = 'https://emsifa.github.io/api-wilayah-indonesia/api'
                    prov = Province.query.get(pid_int)
                    if prov:
                        resp = requests.get(f'{base}/provinces.json', timeout=20)
                        resp.raise_for_status()
                        provinces_json = resp.json() or []
                        api_pid = None
                        for pj in provinces_json:
                            if pj.get('name') == prov.name:
                                api_pid = pj.get('id')
                                break
                        if api_pid:
                            rresp = requests.get(f'{base}/regencies/{api_pid}.json', timeout=20)
                            if rresp.status_code == 200:
                                rjson = rresp.json() or []
                                for rj in rjson:
                                    rname = rj.get('name')
                                    if not rname:
                                        continue
                                    exists = Regency.query.filter_by(name=rname, province_id=pid_int).first()
                                    if not exists:
                                        db.session.add(Regency(name=rname, province_id=pid_int))
                                db.session.commit()
            except Exception:
                db.session.rollback()
                # ignore

        now = datetime.utcnow()
        start_time = now - timedelta(days=days)
        if region_filter:
            chats = ChatHistory.query.join(User, ChatHistory.user_id == User.id) \
                .filter(ChatHistory.created_at >= start_time, User.region == region_filter).all()
        elif province_id:
            # Get all regency names for the selected province
            try:
                from .models import Regency
                pid_int = int(province_id)
                regency_names = [r.name for r in Regency.query.filter_by(province_id=pid_int).all()]
                
                # Debug logging
                print(f"DEBUG topic-distribution: province_id={province_id}, regency_names count={len(regency_names)}")
                if regency_names:
                    print(f"DEBUG topic-distribution: First 5 regency names: {regency_names[:5]}")
                
                if regency_names:
                    # Strategy 1: Exact match with regency names
                    chats = ChatHistory.query.join(User, ChatHistory.user_id == User.id) \
                        .filter(ChatHistory.created_at >= start_time, User.region.in_(regency_names)).all()
                    print(f"DEBUG topic-distribution: Found {len(chats)} chats with exact regency match")
                    
                    # Strategy 2: If no exact match, try fuzzy matching with province name
                    if len(chats) == 0:
                        province = Province.query.get(pid_int)
                        if province:
                            province_name = province.name.lower()
                            # Find users whose region contains province name (fuzzy match)
                            fuzzy_chats = ChatHistory.query.join(User, ChatHistory.user_id == User.id) \
                                .filter(ChatHistory.created_at >= start_time) \
                                .filter(db.func.lower(User.region).contains(province_name)).all()
                            chats = fuzzy_chats
                            print(f"DEBUG topic-distribution: Fuzzy match found {len(chats)} chats for province '{province_name}'")
                else:
                    # If no regencies found, try fallback with province name directly
                    province = Province.query.get(pid_int)
                    if province:
                        province_name = province.name.lower()
                        # Try to find users with region containing province name
                        chats = ChatHistory.query.join(User, ChatHistory.user_id == User.id) \
                            .filter(ChatHistory.created_at >= start_time) \
                            .filter(db.func.lower(User.region).contains(province_name)).all()
                        print(f"DEBUG topic-distribution: Fallback strategy found {len(chats)} chats for province '{province_name}'")
                    else:
                        chats = []
                        print(f"DEBUG topic-distribution: No regencies and no province found for ID {province_id}")
            except ValueError:
                # Invalid province_id, return empty result
                chats = []
                print(f"DEBUG topic-distribution: Invalid province_id: {province_id}")
            except Exception as e:
                # Handle any other database errors
                chats = []
                print(f"DEBUG topic-distribution: Database error for province_id {province_id}: {str(e)}")
        else:
            chats = ChatHistory.query.filter(ChatHistory.created_at >= start_time).all()

        counts_by_user = collections.Counter()
        normalized_topic = topic.strip().lower()
        if not normalized_topic:
            # Aggregate all chats (all topics) per user in the time window
            for c in chats:
                if c.user_id:
                    counts_by_user[c.user_id] += 1
        else:
            for c in chats:
                if not c.user_id:
                    continue
                phrases = [p.lower() for p in extract_phrases(c.question)]
                best = extract_best_phrase(c.question).lower() if c.question else '-'
                if normalized_topic in phrases or normalized_topic == best:
                    counts_by_user[c.user_id] += 1

        if not counts_by_user:
            return jsonify({'markers': []})

        users_query = User.query.filter(
            User.id.in_(list(counts_by_user.keys())),
            User.latitude.isnot(None),
            User.longitude.isnot(None)
        )
        if region_filter:
            users_query = users_query.filter(User.region == region_filter)
        elif province_id:
            from .models import Regency
            users_query = users_query.join(Regency, Regency.name == User.region).filter(Regency.province_id == int(province_id))
        users = users_query.with_entities(User.id, User.latitude, User.longitude, User.region).all()

        markers_map = {}
        for uid, lat, lon, region in users:
            key = (float(lat), float(lon), region or '-')
            markers_map.setdefault(key, 0)
            markers_map[key] += counts_by_user.get(uid, 0)

        # Fallback: if no users have lat/lon, aggregate by region names and geocode centroids
        if not markers_map and chats:
            import os, json as _json
            import requests
            from time import sleep

            cache_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'geocode_cache.json')
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    geo_cache = _json.load(f) or {}
            except Exception:
                geo_cache = {}

            def geocode_region(name: str):
                key = name.strip().upper()
                if key in geo_cache:
                    return geo_cache[key]
                headers = {'User-Agent': 'agri-dashboard/1.0 (contact: admin@example.com)'}
                q = f"{name}, Indonesia"
                url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(q)}&format=json&limit=1&countrycodes=id&accept-language=id"
                try:
                    r = requests.get(url, headers=headers, timeout=12)
                    if r.status_code == 200:
                        arr = r.json() or []
                        if arr:
                            lat = float(arr[0]['lat'])
                            lon = float(arr[0]['lon'])
                            geo_cache[key] = {'lat': lat, 'lon': lon}
                            # Persist lazily
                            try:
                                with open(cache_path, 'w', encoding='utf-8') as fw:
                                    _json.dump(geo_cache, fw)
                            except Exception:
                                pass
                            # polite tiny delay
                            sleep(0.2)
                            return geo_cache[key]
                except Exception:
                    return None
                return None

            # Aggregate counts per region name
            region_counts = {}
            user_map_counts = counts_by_user
            # Build user_id -> region from DB to ensure accurate region mapping
            uid_to_region = {u.id: u.region for u in User.query.filter(User.id.in_(list(user_map_counts.keys()))).all()}
            for uid, cnt in user_map_counts.items():
                reg_name = (uid_to_region.get(uid) or '').strip()
                if not reg_name:
                    continue
                region_counts[reg_name] = region_counts.get(reg_name, 0) + cnt

            for reg_name, cnt in region_counts.items():
                loc = geocode_region(reg_name)
                if not loc:
                    continue
                key = (float(loc['lat']), float(loc['lon']), reg_name)
                markers_map[key] = markers_map.get(key, 0) + cnt

        markers = [
            { 'lat': k[0], 'lon': k[1], 'region': k[2], 'count': v }
            for k, v in markers_map.items()
        ]
        return jsonify({'markers': markers})

    # --- User location update API ---
    @app.route('/api/user/location', methods=['POST'])
    def api_user_location():
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        try:
            if not request.is_json:
                return jsonify({'error': 'Invalid payload'}), 400
            data = request.get_json() or {}
            lat = data.get('lat')
            lon = data.get('lon')
            region = data.get('region')
            if lat is None or lon is None:
                return jsonify({'error': 'lat/lon required'}), 400
            lat = float(lat)
            lon = float(lon)
            user = User.query.get(session.get('user_id'))
            if not user:
                return jsonify({'error': 'User not found'}), 404

            # Try to infer region (kab/kota) via reverse geocoding if missing
            inferred_region = None
            if not region or not str(region).strip():
                try:
                    import requests
                    from .models import Regency
                    headers = {'User-Agent': 'agri-dashboard/1.0 (contact: admin@example.com)'}
                    url = f'https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}&zoom=10&addressdetails=1&accept-language=id'
                    r = requests.get(url, headers=headers, timeout=12)
                    if r.status_code == 200:
                        addr = (r.json() or {}).get('address') or {}
                        candidates = [
                            addr.get('city'), addr.get('county'), addr.get('municipality'),
                            addr.get('town'), addr.get('village')
                        ]
                        candidates = [c for c in candidates if isinstance(c, str) and c.strip()]
                        if candidates:
                            raw = candidates[0].upper()
                            def normalize(n: str) -> str:
                                n = n.upper()
                                for p in ['KABUPATEN ', 'KOTA ADMINISTRASI ', 'KOTA ', 'KAB. ']:
                                    if n.startswith(p):
                                        n = n[len(p):]
                                return n.strip()
                            target = normalize(raw)
                            # Try exact or containment match against known regencies
                            regencies = Regency.query.all()
                            match = None
                            for reg in regencies:
                                rn = normalize(reg.name)
                                if rn == target or rn in target or target in rn:
                                    match = reg.name
                                    break
                            inferred_region = match or candidates[0]
                except Exception:
                    inferred_region = None

            user.latitude = lat
            user.longitude = lon
            # Prefer explicit region; otherwise inferred
            final_region = (region or '').strip() or (inferred_region or '').strip()
            if final_region:
                user.region = final_region
            db.session.commit()
            return jsonify({'ok': True, 'region': user.region})
        except Exception as e:
            logging.exception('[ERROR] updating user location')
            return jsonify({'error': 'Failed to update location'}), 500

    @app.route('/init-db')
    def init_db():
        db.create_all()
        return "✅ Database initialized!"

    @app.route('/drop-db')
    def drop_db():
        db.drop_all()
        return "✅ Semua tabel dihapus."

