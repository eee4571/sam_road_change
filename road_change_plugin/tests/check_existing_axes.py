"""Read-only axis diagnosis on <=200 existing chains; no pipeline or model run."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))


def main():
    project=Path(sys.argv[1]).resolve()
    workspace=next((project/'_work/current/grids/1/periods/20221020/radiometric').glob('*/latest_result.json')).parent
    manifest=json.loads((workspace/'latest_result.json').read_text(encoding='utf8'))
    inputs=json.loads((workspace/'input_manifest.json').read_text(encoding='utf8'))
    run=Path(manifest['run_root']);working=run/'width_review/fast_products.gpkg'
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import LineString
    from engine.road_network_connection import _node_network,_join_chains
    from engine.road_geometry import _RegionalRoadSeed
    from engine.road_axis_cleanup import _axis_candidate,correct_oscillating_chains,remove_exact_duplicates
    from engine.road_connection_evidence import ConnectionEvidence,RoadProbability,probability_sources
    frame=gpd.read_file(working,layer='centerlines')
    projected=frame.estimate_utm_crs() if frame.crs.is_geographic else frame.crs
    frame=frame.to_crs(projected)
    surface=gpd.read_file(working,layer='surfaces').to_crs(projected).geometry.union_all()
    roads=[_RegionalRoadSeed(np.asarray(line.coords),8.,(i,)) for i,line in enumerate(frame.geometry)]
    chains=_join_chains(_node_network(remove_exact_duplicates(roads)))
    evidence=ConnectionEvidence(surface,RoadProbability(probability_sources(Path(inputs['images']),run/'width_review'),projected))
    inspected=0;candidates=0;rejected=0
    for road in sorted(chains,key=lambda r:LineString(r.points).length,reverse=True)[:200]:
        inspected+=1
        candidate=_axis_candidate(road)
        if candidate is None:continue
        candidates+=1
        neighbours=[r for r in chains if r is not road and LineString(r.points).distance(candidate)<1e-5]
        after,count=correct_oscillating_chains([road,*neighbours],evidence)
        if np.array_equal(road.points,after[0].points):
            rejected+=1
            continue
        output=project/'_work/_processing_check/axis_detail'
        output.mkdir(parents=True,exist_ok=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        before=LineString(road.points);corrected=LineString(after[0].points)
        frame=gpd.GeoDataFrame({'state':['before','after']},geometry=[before,corrected],crs=projected)
        frame.to_file(output/'axis_detail.gpkg',layer='axes',driver='GPKG')
        fig,ax=plt.subplots(figsize=(10,5))
        ax.plot(*before.xy,color='#ac7150',lw=1.5,label='Before')
        ax.plot(*corrected.xy,color='#205990',lw=1.2,label='After')
        ax.scatter(*np.asarray(before.coords)[[0,-1]].T,c='black',s=15,zorder=5,label='Fixed endpoints')
        ax.set_aspect('equal');ax.legend();ax.ticklabel_format(useOffset=False,style='plain')
        ax.set_title('Existing 20221020 chain: high-frequency axis correction')
        fig.tight_layout();fig.savefig(output/'axis_detail.png',dpi=160);plt.close(fig)
        result=dict(inspected=inspected,candidates=candidates,evidence_or_topology_rejected=rejected,
                    corrected=1,max_geometric_offset_m=before.hausdorff_distance(corrected),output=str(output))
        (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps(result),flush=True)
        return
    print(json.dumps(dict(inspected=inspected,candidates=candidates,rejected=rejected,corrected=0)),flush=True)


if __name__=='__main__':main()
