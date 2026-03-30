from sqlalchemy import text

class MeasurementRepo:

    @staticmethod
    async def get_exact_customer(customer_ref, timestamp, db):
        query = text("""
            SELECT ms.*
            FROM measurement_summary ms
            JOIN meter m ON ms.serial = m.serial
            WHERE m.customer_ref = :customer_ref
            AND ms.timestamp = :timestamp
        """)
        result = await db.execute(query, {
            "customer_ref": customer_ref,
            "timestamp": timestamp
        })
        return result.mappings().all()

    @staticmethod
    async def get_exact_all(timestamp, db):
        query = text("""
            SELECT * FROM measurement_summary
            WHERE timestamp = :timestamp
        """)
        result = await db.execute(query, {"timestamp": timestamp})
        return result.mappings().all()

    @staticmethod
    async def get_range_customer(customer_ref, start, end, db):
        query = text("""
            SELECT ms.*
            FROM measurement_summary ms
            JOIN meter m ON ms.serial = m.serial
            WHERE m.customer_ref = :customer_ref
            AND ms.timestamp BETWEEN :start AND :end
            ORDER BY ms.timestamp
        """)
        result = await db.execute(query, {
            "customer_ref": customer_ref,
            "start": start,
            "end": end
        })
        return result.mappings().all()

    @staticmethod
    async def get_range_all(start, end, db):
        query = text("""
            SELECT * FROM measurement_summary
            WHERE timestamp BETWEEN :start AND :end
            ORDER BY serial, timestamp
        """)
        result = await db.execute(query, {"start": start, "end": end})
        return result.mappings().all()

    @staticmethod
    async def get_full_customer(customer_ref, db):
        query = text("""
            SELECT ms.*
            FROM measurement_summary ms
            JOIN meter m ON ms.serial = m.serial
            WHERE m.customer_ref = :customer_ref
            ORDER BY ms.timestamp
        """)
        result = await db.execute(query, {"customer_ref": customer_ref})
        return result.mappings().all()