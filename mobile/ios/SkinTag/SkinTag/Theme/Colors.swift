//
//  Colors.swift
//  SkinTag
//
//  Standardized color theme matching web app design
//

import SwiftUI

extension Color {
    // MARK: - Brand Colors

    /// Primary background cream color
    static let cream = Color(hex: "#FAF7F2")

    /// Warm white for cards and elevated surfaces
    static let warmWhite = Color(hex: "#FFFCF7")

    /// Primary sage green
    static let sage = Color(hex: "#8B9D83")

    /// Light sage for subtle backgrounds
    static let sageLight = Color(hex: "#C4D1BE")

    /// Dark sage for text and emphasis
    static let sageDark = Color(hex: "#5A6B54")

    /// Accent terracotta
    static let terracotta = Color(hex: "#C4704B")

    /// Light terracotta for highlights
    static let terracottaLight = Color(hex: "#E8A889")

    /// Primary text color
    static let charcoal = Color(hex: "#2C2C2C")

    /// Secondary text color
    static let charcoalLight = Color(hex: "#4A4A4A")

    // MARK: - Risk Colors

    /// Low risk - green
    static let riskLow = Color(hex: "#7BA05B")

    /// Medium risk - amber
    static let riskMedium = Color(hex: "#E5A84B")

    /// High risk - red/coral
    static let riskHigh = Color(hex: "#D4644B")

    // MARK: - Utility Colors

    /// Trust/info blue
    static let trustBlue = Color(hex: "#5B8FA0")

    // MARK: - Hex Initializer

    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (1, 1, 1, 0)
        }

        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

// MARK: - Theme Convenience

struct AppTheme {
    // Backgrounds
    static let background = Color.cream
    static let surface = Color.warmWhite
    static let surfaceElevated = Color.white

    // Text
    static let textPrimary = Color.charcoal
    static let textSecondary = Color.charcoalLight

    // Primary
    static let primary = Color.sage
    static let primaryLight = Color.sageLight
    static let primaryDark = Color.sageDark

    // Accent
    static let accent = Color.terracotta
    static let accentLight = Color.terracottaLight

    // Semantic
    static let success = Color.riskLow
    static let warning = Color.riskMedium
    static let danger = Color.riskHigh
    static let info = Color.trustBlue

    // Risk levels
    static func riskColor(for probability: Double) -> Color {
        if probability < 0.3 {
            return .riskLow
        } else if probability < 0.7 {
            return .riskMedium
        } else {
            return .riskHigh
        }
    }
}
