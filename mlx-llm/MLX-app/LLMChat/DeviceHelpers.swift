//
//  DeviceHelpers.swift
//  LLMChat
//
//  Created by MD Sahil AK on 11/02/26.
//

import Foundation

#if canImport(IOKit)
import IOKit.ps
#endif

#if canImport(UIKit)
import UIKit
#endif

/// Cross-platform utilities for reading device battery and thermal information.
///
/// Uses IOKit on macOS and UIKit on iOS. Returns sentinel values (`-1` or `"UNKNOWN"`)
/// on unsupported platforms.
enum DeviceHelpers {

    /// Returns the current battery charge percentage (0–100), or `-1` if unavailable.
    static func getBatteryLevel() -> Int {
#if os(macOS)
        let snapshot = IOPSCopyPowerSourcesInfo().takeRetainedValue()
        let sources = IOPSCopyPowerSourcesList(snapshot).takeRetainedValue() as [CFTypeRef]
        
        for source in sources {
            if let info = IOPSGetPowerSourceDescription(snapshot, source).takeUnretainedValue() as? [String: Any],
               let capacity = info[kIOPSCurrentCapacityKey] as? Int {
                return capacity
            }
        }
        
        return -1
        
#elseif os(iOS)
        
        UIDevice.current.isBatteryMonitoringEnabled = true
        let level = UIDevice.current.batteryLevel
        return level >= 0 ? Int(level * 100) : -1
        
#else
        
        return -1
        
#endif
    }
    
    /// Returns a human-readable string describing the device's current thermal state.
    ///
    /// Possible values: `"Normal"`, `"Warm!"`, `"Hot!!"`, `"Very Hot!!!"`, or `"UNKNOWN"`.
    static func getBatteryThermalState() -> String {
        let state = ProcessInfo.processInfo.thermalState
        
        switch state {
        case .nominal:
            return "Normal"
        case .fair:
            return "Warm!"
        case .serious:
            return "Hot!!"
        case .critical:
            return "Very Hot!!!"
            
        @unknown default:
            return "UNKNOWN"
        }
    }
}
