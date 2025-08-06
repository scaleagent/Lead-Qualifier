#!/usr/bin/env python3
import asyncio
from datetime import datetime, timedelta

from repos.database import AsyncSessionLocal
from repos.contractor_repo import ContractorRepo
from repos.conversation_repo import ConversationRepo
from repos.conversation_data_repo import ConversationDataRepo


async def seed():
    async with AsyncSessionLocal() as session:
        contractor_repo = ContractorRepo(session)
        conv_repo = ConversationRepo(session)
        data_repo = ConversationDataRepo(session)

        # ——— Helper to get-or-create a contractor ———
        async def get_or_create_contractor(name, phone, config):
            existing = await contractor_repo.get_by_phone(phone)
            if existing:
                print(f"ℹ️ Contractor {phone} already exists, loading")
                contractor = existing
            else:
                contractor = await contractor_repo.create(name=name,
                                                          phone=phone)
                print(f"➕ Created contractor {phone}")
            # Always overwrite digest_config
            contractor.digest_config = config
            session.add(contractor)
            await session.commit()
            return contractor

        # 1) Create/get two contractors
        c1 = await get_or_create_contractor(name="Alice Home",
                                            phone="+441234567890",
                                            config={
                                                "digest_hour": 18,
                                                "timezone": "Europe/London",
                                                "repeat_until_takeover": True
                                            })
        c2 = await get_or_create_contractor(name="Bob Builders",
                                            phone="+441234567891",
                                            config={
                                                "digest_hour": 17,
                                                "timezone": "Europe/London",
                                                "repeat_until_takeover": False
                                            })

        # 2) Seed leads for Alice
        # (same as before, but you can wrap each in get-or-create if needed)

        # 2a) Fresh qualified lead
        convo_a1 = await conv_repo.create_conversation(
            contractor_id=c1.id, customer_phone="+447700900001")
        await data_repo.upsert(conversation_id=convo_a1.id,
                               contractor_id=c1.id,
                               customer_phone="+447700900001",
                               data_dict={"job_type": "Kitchen Renovation"},
                               qualified=True,
                               job_title="Kitchen Renovation")

        # 2b) Opted-out lead
        convo_a2 = await conv_repo.create_conversation(
            contractor_id=c1.id, customer_phone="+447700900002")
        await data_repo.upsert(conversation_id=convo_a2.id,
                               contractor_id=c1.id,
                               customer_phone="+447700900002",
                               data_dict={"job_type": "Bathroom Remodel"},
                               qualified=True,
                               job_title="Bathroom Remodel")
        await data_repo.mark_digest_opt_out(convo_a2.id)

        # 2c) Ongoing note-collection lead
        convo_a3 = await conv_repo.create_conversation(
            contractor_id=c1.id, customer_phone="+447700900003")
        convo_a3.status = "COLLECTING_NOTES"
        session.add(convo_a3)
        await session.commit()
        await data_repo.upsert(conversation_id=convo_a3.id,
                               contractor_id=c1.id,
                               customer_phone="+447700900003",
                               data_dict={"job_type": "Roof Repair"},
                               qualified=True,
                               job_title="Roof Repair")

        # 3) Seed leads for Bob
        convo_b1 = await conv_repo.create_conversation(
            contractor_id=c2.id, customer_phone="+447700900010")
        await data_repo.upsert(conversation_id=convo_b1.id,
                               contractor_id=c2.id,
                               customer_phone="+447700900010",
                               data_dict={"job_type": "Exterior Painting"},
                               qualified=True,
                               job_title="Exterior Painting")
        # mark as already sent yesterday
        yesterday = datetime.utcnow() - timedelta(days=1)
        await data_repo.mark_digest_sent(convo_b1.id, yesterday)

    print("✅ Seed data inserted successfully.")


if __name__ == "__main__":
    asyncio.run(seed())
