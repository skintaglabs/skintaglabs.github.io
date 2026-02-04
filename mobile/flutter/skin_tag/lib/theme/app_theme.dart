import 'package:flutter/material.dart';

/// SkinTag app color palette - matching web app design
class AppColors {
  // Brand colors
  static const Color cream = Color(0xFFFAF7F2);
  static const Color warmWhite = Color(0xFFFFFCF7);
  static const Color sage = Color(0xFF8B9D83);
  static const Color sageLight = Color(0xFFC4D1BE);
  static const Color sageDark = Color(0xFF5A6B54);
  static const Color terracotta = Color(0xFFC4704B);
  static const Color terracottaLight = Color(0xFFE8A889);
  static const Color charcoal = Color(0xFF2C2C2C);
  static const Color charcoalLight = Color(0xFF4A4A4A);

  // Risk colors
  static const Color riskLow = Color(0xFF7BA05B);
  static const Color riskMedium = Color(0xFFE5A84B);
  static const Color riskHigh = Color(0xFFD4644B);

  // Utility
  static const Color trustBlue = Color(0xFF5B8FA0);

  /// Get risk color based on probability
  static Color getRiskColor(double probability) {
    if (probability < 0.3) return riskLow;
    if (probability < 0.7) return riskMedium;
    return riskHigh;
  }
}

/// App theme configuration
class AppTheme {
  static ThemeData get lightTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      scaffoldBackgroundColor: AppColors.cream,
      colorScheme: const ColorScheme.light(
        primary: AppColors.sage,
        primaryContainer: AppColors.sageLight,
        secondary: AppColors.terracotta,
        secondaryContainer: AppColors.terracottaLight,
        surface: AppColors.warmWhite,
        surfaceContainerHighest: AppColors.cream,
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onSurface: AppColors.charcoal,
        error: AppColors.riskHigh,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: AppColors.cream,
        foregroundColor: AppColors.charcoal,
        elevation: 0,
        centerTitle: true,
      ),
      cardTheme: CardTheme(
        color: AppColors.warmWhite,
        elevation: 2,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.sage,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: AppColors.sage,
        ),
      ),
      textTheme: const TextTheme(
        headlineLarge: TextStyle(
          color: AppColors.charcoal,
          fontWeight: FontWeight.bold,
        ),
        headlineMedium: TextStyle(
          color: AppColors.charcoal,
          fontWeight: FontWeight.bold,
        ),
        bodyLarge: TextStyle(
          color: AppColors.charcoal,
        ),
        bodyMedium: TextStyle(
          color: AppColors.charcoalLight,
        ),
      ),
    );
  }

  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AppColors.charcoal,
      colorScheme: const ColorScheme.dark(
        primary: AppColors.sage,
        primaryContainer: AppColors.sageDark,
        secondary: AppColors.terracotta,
        secondaryContainer: AppColors.terracottaLight,
        surface: Color(0xFF3A3A3A),
        onPrimary: Colors.white,
        onSecondary: Colors.white,
        onSurface: AppColors.cream,
        error: AppColors.riskHigh,
      ),
    );
  }
}
