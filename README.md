# Network anomaly detection

## Metodología original

![figures/final_base_flux.png](figures/final_base_flux.png)

## Metodología modificada para mejorar rendimiento

![figures/final_flux_wmods.png](figures/final_flux_wmods.png)

## Instalación
Además de clonar el respositorio, es necesario descargar el _dataset_ para poder ejecutar los _scripts_. Para ello podemos usar la siguiente secuencia de comandos desde dentro del respositorio:

### En sistemas Linux:
```bash
cd data
wget -O dataset_dist.csv "https://udcgal-my.sharepoint.com/:x:/g/personal/julio_jairo_estevez_pereira_udc_es/EQWdSJToODxJhYe9gaWyV0MBWI19AxHXivB1y4-DrP4Myg?e=9fzMs6&download=1"
````
### En sistemas Windows (desde Powershell):
```bash
cd data
wget "https://udcgal-my.sharepoint.com/:x:/g/personal/julio_jairo_estevez_pereira_udc_es/EQWdSJToODxJhYe9gaWyV0MBWI19AxHXivB1y4-DrP4Myg?e=9fzMs6&download=1" -OutFile dataset_dist.csv
````
