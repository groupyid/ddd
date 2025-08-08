#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routes.models import db, User, ChatHistory, Province, Regency
from app import app
from datetime import datetime, timedelta
import random

def create_dummy_data():
    with app.app_context():
        print("=== CREATING DUMMY DATA ===\n")
        
        # Data topik dan pertanyaan
        topics_data = {
            "menanam padi": [
                "Bagaimana cara menanam padi yang baik di musim hujan?",
                "Kapan waktu terbaik untuk menanam padi di sawah?",
                "Apa saja varietas padi yang cocok untuk lahan basah?"
            ],
            "pola penanaman": [
                "Bagaimana pola penanaman yang efektif untuk hasil maksimal?",
                "Apa itu sistem tanam jajar legowo dan bagaimana penerapannya?"
            ],
            "penanaman nilam": [
                "Bagaimana cara menanam nilam di lahan kering?",
                "Apa syarat tumbuh tanaman nilam yang optimal?"
            ],
            "pola penanaman nilam": [
                "Bagaimana jarak tanam nilam yang ideal?",
                "Apa pola penanaman nilam untuk hasil minyak terbaik?"
            ],
            "mengatasi hama": [
                "Bagaimana cara mengatasi hama wereng pada tanaman padi?",
                "Apa pestisida alami yang efektif untuk mengatasi hama?"
            ]
        }
        
        # Data daerah dengan koordinat
        regions_data = [
            {"name": "Banda Aceh", "lat": 5.5483, "lon": 95.3238, "province": "Aceh"},
            {"name": "Medan", "lat": 3.5952, "lon": 98.6722, "province": "Sumatera Utara"},
            {"name": "Padang", "lat": -0.9471, "lon": 100.4172, "province": "Sumatera Barat"},
            {"name": "Palembang", "lat": -2.9761, "lon": 104.7754, "province": "Sumatera Selatan"},
            {"name": "Jakarta", "lat": -6.2088, "lon": 106.8456, "province": "DKI Jakarta"},
            {"name": "Bandung", "lat": -6.9175, "lon": 107.6191, "province": "Jawa Barat"},
            {"name": "Semarang", "lat": -6.9667, "lon": 110.4167, "province": "Jawa Tengah"},
            {"name": "Surabaya", "lat": -7.2575, "lon": 112.7521, "province": "Jawa Timur"},
            {"name": "Denpasar", "lat": -8.6500, "lon": 115.2167, "province": "Bali"},
            {"name": "Makassar", "lat": -5.1477, "lon": 119.4327, "province": "Sulawesi Selatan"}
        ]
        
        # Ensure provinces exist
        for region_data in regions_data:
            province_name = region_data["province"]
            province = Province.query.filter_by(name=province_name).first()
            if not province:
                province = Province(name=province_name)
                db.session.add(province)
                print(f"Added province: {province_name}")
        
        db.session.commit()
        
        # Create users for each region
        users_created = []
        user_id_counter = 1000  # Start from 1000 to avoid conflicts
        
        for i, region_data in enumerate(regions_data):
            # Create 2-3 users per region
            for j in range(2 + (i % 2)):  # 2 or 3 users per region
                user_id_counter += 1
                
                # Check if user already exists
                existing_user = User.query.get(user_id_counter)
                if existing_user:
                    continue
                
                user = User(
                    id=user_id_counter,
                    name=f"Petani_{region_data['name']}_{j+1}",
                    dob=datetime(1990 + (i % 10), 1 + (j % 12), 1 + (i % 28)),
                    gender="Laki-laki" if j % 2 == 0 else "Perempuan",
                    email=f"petani_{region_data['name'].lower()}_{j+1}@example.com",
                    phone=f"08123456{i:02d}{j}",
                    password="hashed_password",
                    role="petani",
                    region=region_data["name"],
                    latitude=region_data["lat"] + random.uniform(-0.1, 0.1),  # Add some variation
                    longitude=region_data["lon"] + random.uniform(-0.1, 0.1),
                    created_at=datetime.utcnow() - timedelta(days=random.randint(30, 365))
                )
                
                db.session.add(user)
                users_created.append(user)
                print(f"Created user: {user.name} in {region_data['name']}")
        
        db.session.commit()
        print(f"\nCreated {len(users_created)} users")
        
        # Create chat history
        chat_id_counter = 2000  # Start from 2000 to avoid conflicts
        chats_created = []
        
        # Distribute questions across topics and users
        all_users = users_created
        
        for topic, questions in topics_data.items():
            print(f"\nCreating chats for topic: {topic}")
            
            for question in questions:
                chat_id_counter += 1
                
                # Select random user
                user = random.choice(all_users)
                
                # Create chat at random time in last 30 days
                created_at = datetime.utcnow() - timedelta(
                    days=random.randint(1, 30),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
                
                # Generate answer based on topic
                answers = {
                    "menanam padi": "Untuk menanam padi yang baik, pastikan lahan sudah disiapkan dengan pengolahan tanah yang tepat. Gunakan benih unggul dan lakukan penanaman dengan jarak yang sesuai.",
                    "pola penanaman": "Pola penanaman yang efektif adalah dengan sistem jajar legowo atau sistem berbaris dengan jarak yang teratur untuk memaksimalkan sinar matahari dan sirkulasi udara.",
                    "penanaman nilam": "Nilam dapat ditanam dengan stek batang. Pilih lahan yang tidak tergenang air dan memiliki drainase yang baik. Jarak tanam ideal adalah 50x50 cm.",
                    "pola penanaman nilam": "Untuk hasil minyak optimal, gunakan jarak tanam 50x50 cm dengan pola berbaris. Lakukan pemangkasan rutin untuk merangsang pertumbuhan daun.",
                    "mengatasi hama": "Gunakan pestisida nabati seperti ekstrak neem atau bawang putih. Lakukan pemantauan rutin dan terapkan sistem pengendalian hama terpadu (PHT)."
                }
                
                answer = answers.get(topic, "Terima kasih atas pertanyaannya. Tim ahli kami akan memberikan solusi terbaik untuk masalah pertanian Anda.")
                
                chat = ChatHistory(
                    id=chat_id_counter,
                    user_id=user.id,
                    question=question,
                    answer=answer,
                    sources="Database Pertanian Indonesia, Panduan Bertani Modern",
                    created_at=created_at
                )
                
                db.session.add(chat)
                chats_created.append(chat)
                print(f"  - {question[:50]}... (User: {user.name}, Region: {user.region})")
        
        db.session.commit()
        print(f"\nCreated {len(chats_created)} chat histories")
        
        # Summary
        print("\n=== SUMMARY ===")
        print(f"Total users created: {len(users_created)}")
        print(f"Total chats created: {len(chats_created)}")
        print("\nTopics distribution:")
        for topic, questions in topics_data.items():
            print(f"  - {topic}: {len(questions)} pertanyaan")
        
        print("\nRegions with users:")
        for region_data in regions_data:
            user_count = len([u for u in users_created if u.region == region_data["name"]])
            print(f"  - {region_data['name']} ({region_data['province']}): {user_count} users")
        
        print("\n✅ Dummy data generation completed!")

if __name__ == "__main__":
    create_dummy_data()