from flask import request, render_template, jsonify, session, redirect, url_for
from werkzeug.security import check_password_hash, generate_password_hash
import logging
from .models import db, User
from .rag_core import get_rag_response, initialize_rag_system
import time
from datetime import datetime
from uuid import uuid4
import json



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
        # Early Warning System: deteksi topik yang sering muncul di wilayah tertentu
        # Threshold: minimal 3 user berbeda membahas topik sama di wilayah sama dalam 7 hari
        ews = []
        # Buat mapping: {(region, topik): set(user_id), count}
        ews_map = {}
        for c in chats:
            user = user_map.get(c.user_id)
            region = user.region if user and user.region else '-'
            topik = extract_topik(c.question)
            key = (region, topik)
            if key not in ews_map:
                ews_map[key] = {'users': set(), 'count': 0}
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
        if 'user_id' not in session or session.get('user_role') != 'admin':
            return redirect(url_for('login'))
        from .models import ChatHistory, User
        from sqlalchemy import func
        from datetime import datetime, timedelta
        import collections, re

        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        chats = ChatHistory.query.filter(ChatHistory.created_at >= week_ago).all()

        jumlah_pertanyaan = len(chats)
        all_questions = ' '.join([c.question for c in chats]).lower()
        stopwords = set(['dan','atau','yang','di','ke','dari','untuk','pada','dengan','apa','bagaimana','adalah','saya','ini','itu','the','of','in','to','a','is','are','it','as','by','an','be','on','for','was','were','has','have','had','will','can','petani','chatbot'])
        keywords = re.findall(r'\b\w+\b', all_questions)
        keywords = [w for w in keywords if w not in stopwords and len(w) > 2]
        counter = collections.Counter(keywords)
        top_topics = counter.most_common(5)

        # All region & topic for filter
        all_regions = sorted(set(u.region for u in User.query.filter(User.region != None)))
        all_topics = sorted(set([t[0] for t in counter.most_common(15)]))

        # All chats for table (join user)
        user_map = {u.id: u for u in User.query.all()}
        def extract_topik(q):
            # Ambil kata kunci/topik pertama yang bukan stopword
            for w in re.findall(r'\b\w+\b', (q or '').lower()):
                if w not in stopwords and len(w) > 2:
                    return w
            return '-'
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
        jam_counter = collections.Counter([c.created_at.hour for c in chats])
        jam_aktif_tertinggi = f"{jam_counter.most_common(1)[0][0]}:00" if jam_counter else '-'
        user_counter = collections.Counter([c.user_id for c in chats])
        rata2_pertanyaan_per_user = f"{(jumlah_pertanyaan/len(user_counter)):.2f}" if user_counter else '-'
        user_paling_aktif = user_map[user_counter.most_common(1)[0][0]].name if user_counter else '-'

        # Lokasi untuk peta
        user_ids = set(c.user_id for c in chats)
        locations = User.query.filter(User.id.in_(user_ids), User.latitude != None, User.longitude != None).with_entities(User.latitude, User.longitude, User.region, User.name).all()
        locations_json = [
            {"lat": float(lat), "lon": float(lon), "region": region, "name": name}
            for lat, lon, region, name in locations
        ]
        wilayah_aktif = ', '.join(sorted(set(l["region"] for l in locations_json if l["region"])) ) if locations_json else '-'

        return render_template('dashboard_admin.html',
            user=session.get('user_name'),
            jumlah_pertanyaan=jumlah_pertanyaan,
            top_topics=top_topics,
            wilayah_aktif=wilayah_aktif,
            locations_json=locations_json,
            all_regions=all_regions,
            all_topics=all_topics,
            all_chats=all_chats,
            jam_aktif_tertinggi=jam_aktif_tertinggi,
            rata2_pertanyaan_per_user=rata2_pertanyaan_per_user,
            user_paling_aktif=user_paling_aktif,
            ews=ews
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
                'processing_time': round(processing_time, 2)
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


    @app.route('/init-db')
    def init_db():
        db.create_all()
        return "✅ Database initialized!"

    @app.route('/drop-db')
    def drop_db():
        db.drop_all()
        return "✅ Semua tabel dihapus."

