"""
Script to extract target data for multiple curves with specific tags.
This helps gather data for different price points (min, max, wavg, percentiles).
"""
import os
import argparse
import pandas as pd
import volue_insight_timeseries
from curves import curve_collections

def get_curve_tags(curve_name, session):
    """
    Get available tags for a curve.
    
    Args:
        curve_name: Name of the curve
        session: API session
        
    Returns:
        List of available tags
    """
    try:
        curve = session.get_curve(name=curve_name)
        if curve is None:
            print(f"Warning: Curve {curve_name} not found")
            return []
        
        tags = curve.get_tags()
        return tags
    except Exception as e:
        print(f"Error getting tags for curve {curve_name}: {str(e)}")
        return []

def extract_curve_data(curve_name, session, start_date, end_date, 
                      tags=None, output_dir="./src/data/extracted"):
    """
    Extract data for a curve with multiple tags.
    
    Args:
        curve_name: Name of the curve
        session: API session
        start_date: Start date for data extraction
        end_date: End date for data extraction
        tags: List of tags to extract (None = all available tags)
        output_dir: Directory to store extracted data
        
    Returns:
        Dictionary with extraction results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    curve = session.get_curve(name=curve_name)
    if curve is None:
        return {
            "curve": curve_name,
            "error": "Curve not found",
            "extracted_tags": 0
        }
    
    # Get available tags
    available_tags = curve.get_tags()
    print(f"\nAvailable tags for curve {curve_name}:")
    for i, tag in enumerate(available_tags):
        print(f"  {i+1}. {tag}")
    
    # Determine which tags to extract
    if tags is None:
        # Extract all tags
        tags_to_extract = available_tags
    elif isinstance(tags, list):
        # Extract specified tags if they exist
        tags_to_extract = [tag for tag in tags if tag in available_tags]
    else:
        # Extract a single tag if it exists
        tags_to_extract = [tags] if tags in available_tags else []
    
    if not tags_to_extract:
        return {
            "curve": curve_name,
            "error": "No valid tags to extract",
            "available_tags": available_tags,
            "extracted_tags": 0
        }
    
    # Extract data for each tag
    results = []
    
    for tag in tags_to_extract:
        try:
            print(f"Extracting data for {curve_name} with tag {tag}...")
            
            # Get data with the specified tag
            ts = curve.get_data(tag=tag, data_from=start_date, data_to=end_date)
            
            # Convert to pandas Series
            s = ts.to_pandas()
            
            # Convert to DataFrame
            df = pd.DataFrame({f"{curve_name}_{tag}": s})
            
            # Save to CSV
            output_file = os.path.join(output_dir, f"{curve_name[:18]}_{tag}.csv")
            df.to_csv(output_file)
            
            print(f"  Extracted {len(df)} rows, saved to {output_file}")
            
            results.append({
                "tag": tag,
                "rows": len(df),
                "file": output_file
            })
            
        except Exception as e:
            print(f"  Error extracting {curve_name} with tag {tag}: {str(e)}")
            results.append({
                "tag": tag,
                "error": str(e)
            })
    
    return {
        "curve": curve_name,
        "results": results,
        "extracted_tags": len(results)
    }

def extract_target_prices(area, session, start_date, end_date,
                         target_type=None, tags=None, 
                         output_dir="./src/data/extracted"):
    """
    Extract imbalance price data for an area.
    
    Args:
        area: Area code (e.g., 'de')
        session: API session
        start_date: Start date for data extraction
        end_date: End date for data extraction
        target_type: Type of target to extract ('mfrr', 'afrr', or None for both)
        tags: List of tags to extract (None = all available)
        output_dir: Directory to store extracted data
        
    Returns:
        Dictionary with extraction results
    """
    # Get target curves for the area
    if area in curve_collections:
        area_curves = curve_collections[area]
    else:
        print(f"No curves defined for area: {area}")
        return {"error": f"No curves defined for area: {area}"}
    
    # Determine which target types to extract
    target_curves = []
    
    if target_type is None or target_type == 'mfrr':
        if 'mfrr' in area_curves:
            target_curves.extend(area_curves['mfrr'])
    
    if target_type is None or target_type == 'afrr':
        if 'afrr' in area_curves:
            target_curves.extend(area_curves['afrr'])
    
    if not target_curves:
        print(f"No target curves found for area {area} and target type {target_type}")
        return {"error": f"No target curves found for area {area} and target type {target_type}"}
    
    # Extract data for each target curve
    results = []
    
    for curve_name in target_curves:
        result = extract_curve_data(
            curve_name=curve_name,
            session=session,
            start_date=start_date,
            end_date=end_date,
            tags=["50pct"],
            output_dir=output_dir
        )
        
        results.append(result)
    
    return {
        "area": area,
        "target_type": target_type,
        "processed_curves": len(results),
        "results": results
    }

def main():
    parser = argparse.ArgumentParser(description='Extract target data with specific tags')
    
    parser.add_argument('--area', type=str, default='de',
                       help='Area code (default: de)')
    
    parser.add_argument('--type', type=str, default=None,
                       help='Target type to extract (mfrr, afrr, or None for both)')
    
    parser.add_argument('--tag', type=str, default=None, nargs='+',
                       help='Tag(s) to extract (default: all available tags)')
    
    parser.add_argument('--common-tags', action='store_true',
                       help='Extract only common tags like min, max, wavg, 25pct, 50pct, 75pct')
    
    parser.add_argument('--start-date', type=str, default='2021-01-01',
                       help='Start date for data extraction (default: 2021-01-01)')
    
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for data extraction (default: today)')
    
    parser.add_argument('--output', type=str, default='./src/data/extracted',
                       help='Directory to store extracted data (default: ./src/data/extracted)')
    
    parser.add_argument('--list-tags', action='store_true',
                       help='List available tags for the curves and exit')
    
    args = parser.parse_args()
    
    # Initialize API session
    session = volue_insight_timeseries.Session(config_file=os.environ.get("WAPI_CONFIG"))
    
    # Set up dates
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp.now()
    
    # Get target curves for the area
    if args.area in curve_collections:
        area_curves = curve_collections[args.area]
        mfrr_targets = area_curves.get('mfrr', [])
        afrr_targets = area_curves.get('afrr', [])
        all_targets = mfrr_targets + afrr_targets
    else:
        print(f"No curves defined for area: {args.area}")
        return
    
    if args.list_tags:
        print(f"\nListing available tags for {args.area} target curves:")
        
        for curve_name in all_targets:
            tags = get_curve_tags(curve_name, session)
            print(f"\n{curve_name}:")
            if tags:
                for i, tag in enumerate(tags):
                    print(f"  {i+1}. {tag}")
            else:
                print("  No tags found")
        
        return
    
    # Determine tags to extract
    if args.common_tags:
        tags_to_extract = ['min', 'max', 'wavg', '25pct', '50pct', '75pct']
    else:
        tags_to_extract = args.tag
    
    print(f"\nExtracting data for {args.area} target curves")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Tags to extract: {tags_to_extract if tags_to_extract else 'all available'}")
    
    # Extract data
    results = extract_target_prices(
        area=args.area,
        session=session,
        start_date=start_date,
        end_date=end_date,
        target_type=args.type,
        tags=tags_to_extract,
        output_dir=args.output
    )
    
    print("\nExtraction complete")
    print(f"Processed {results.get('processed_curves', 0)} curves")
    
    # Print summary of extracted data
    for curve_result in results.get('results', []):
        curve_name = curve_result.get('curve')
        
        if 'results' in curve_result:
            successful = sum(1 for r in curve_result['results'] if 'error' not in r)
            print(f"  {curve_name}: {successful}/{len(curve_result['results'])} tags extracted")
        else:
            print(f"  {curve_name}: Error - {curve_result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()