from ChestCancerClassification.config.configuration import ConfigurationManager
from ChestCancerClassification.entity.config_entity import DataIngestionConfig
from ChestCancerClassification.components.data_ingestion import DataIngestion
from ChestCancerClassification import logger

STAGE_NAME = "Data Ingestion stage"


class DataIngestionPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
