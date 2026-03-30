from sqlalchemy import text

class PredictionRepo:

    @staticmethod
    async def get_exact(customer_ref, timestamp, db):
        query = text("""
            SELECT * FROM customer_prediction
            WHERE customer_ref = :customer_ref
            AND prediction_timestamp = :timestamp
        """)
        result = await db.execute(query, {
            "customer_ref": customer_ref,
            "timestamp": timestamp
        })
        return result.mappings().all()

    @staticmethod
    async def get_all(timestamp, db):
        query = text("""
            SELECT * FROM customer_prediction
            WHERE prediction_timestamp = :timestamp
        """)
        result = await db.execute(query, {"timestamp": timestamp})
        return result.mappings().all()

    @staticmethod
    async def get_range_customer(customer_ref, start, end, db):
        query = text("""
            SELECT * FROM customer_prediction
            WHERE customer_ref = :customer_ref
            AND prediction_timestamp BETWEEN :start AND :end
            ORDER BY prediction_timestamp
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
            SELECT * FROM customer_prediction
            WHERE prediction_timestamp BETWEEN :start AND :end
            ORDER BY customer_ref, prediction_timestamp
        """)
        result = await db.execute(query, {"start": start, "end": end})
        return result.mappings().all()