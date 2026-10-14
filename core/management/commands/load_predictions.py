from django.core.management.base import BaseCommand
from core.models import StockPrediction
import pandas as pd
from pathlib import Path


class Command(BaseCommand):
    help = 'Carga predicciones del modelo desde CSV a la base de datos'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            default='data/predictions.csv',
            help='Ruta al archivo CSV de predicciones'
        )

    def handle(self, *args, **options):
        file_path = Path(options['file'])
        
        if not file_path.exists():
            self.stdout.write(
                self.style.ERROR(f'Archivo no encontrado: {file_path}')
            )
            return
        
        self.stdout.write(f'Cargando predicciones desde {file_path}...')
        
        df = pd.read_csv(file_path)
        self.stdout.write(f'Leídas {len(df)} filas del CSV')
        
        # Limpiar datos existentes
        StockPrediction.objects.all().delete()
        self.stdout.write('Base de datos limpiada')
        
        # Crear predicciones en bulk
        predictions = []
        for _, row in df.iterrows():
            predictions.append(
                StockPrediction(
                    ticker=row['instrument'],
                    date=row['datetime'],
                    predicted_return=row['prediction'],
                    actual_return=row.get('actual', None)
                )
            )
        
        StockPrediction.objects.bulk_create(predictions)
        self.stdout.write(
            self.style.SUCCESS(f'Creadas {len(predictions)} predicciones')
        )
