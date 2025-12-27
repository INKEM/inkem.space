// Skill data configuration file
// Used to manage data for the skill display page

export interface Skill {
	id: string;
	name: string;
	description: string;
	icon: string; // Iconify icon name
	category: "frontend" | "后端开发" | "database" | "软件工具" | "other" | "硬件开发";
	level: "beginner" | "intermediate" | "advanced" | "expert";
	experience: {
		years: number;
		months: number;
	};
	projects?: string[]; // Related project IDs
	certifications?: string[];
	color?: string; // Skill card theme color
}

export const skillsData: Skill[] = [
	{
		id: "python",
		name: "Python",
		description:
			"Python编程语言，广泛应用于数据科学、机器学习和Web开发。",
		icon: "logos:python",
		category: "后端开发",
		level: "intermediate",
		experience: { years: 2, months: 3 },
		projects: [],
		color: "#eeff00ff",
	},
	{
		id: "cpp",
		name: "C++",
		description:
			"C++编程语言，广泛应用于系统编程和高性能计算。",
		icon: "logos:c-plusplus",
		category: "后端开发",
		level: "intermediate",
		experience: { years: 2, months: 3 },
		projects: [],
		color: "#6093f2ff",
	},
	{
		id: "STM32",
		name: "STM32",
		description:
			"STM32微控制器系列，广泛应用于嵌入式系统和物联网项目。",
		icon: "simple-icons:stmicroelectronics",
		category: "硬件开发",
		level: "intermediate",
		experience: { years: 0, months: 9 },
		projects: [],
		color: "#1e135aff",
	},
	{
		id: "solidworks",
		name: "SolidWorks",
		description:
			"3D CAD设计软件，广泛应用于机械设计和产品开发领域。",
		icon: "charm:cube",
		category: "软件工具",
		level: "advanced",
		experience: { years: 1, months: 3 },
		projects: [],
		color: "#590f09ff",
	},
	{
		id: "vscode",
		name: "VS Code",
		description:
			"轻量级但功能强大的源代码编辑器，支持多种编程语言和插件扩展。",
		icon: "logos:visual-studio-code",
		category: "软件工具",
		level: "intermediate",
		experience: { years: 2, months: 3 },
		projects: [],
		color: "#4acbfaff",
	},
	{
		id: "autocad",
		name: "AutoCAD",
		description:
			"2D/3D CAD设计软件，广泛应用于建筑和工程设计领域。",
		icon: "simple-icons:autocad",
		category: "软件工具",
		level: "intermediate",
		experience: { years: 0, months: 6 },
		projects: [],
		color: "#b5154aff",
	},
];

// Get skill statistics
export const getSkillStats = () => {
	const total = skillsData.length;
	const byLevel = {
		beginner: skillsData.filter((s) => s.level === "beginner").length,
		intermediate: skillsData.filter((s) => s.level === "intermediate")
			.length,
		advanced: skillsData.filter((s) => s.level === "advanced").length,
		expert: skillsData.filter((s) => s.level === "expert").length,
	};
	const byCategory = {
		frontend: skillsData.filter((s) => s.category === "frontend").length,
		backend: skillsData.filter((s) => s.category === "backend").length,
		database: skillsData.filter((s) => s.category === "database").length,
		tools: skillsData.filter((s) => s.category === "tools").length,
		other: skillsData.filter((s) => s.category === "other").length,
	};

	return { total, byLevel, byCategory };
};

// Get skills by category
export const getSkillsByCategory = (category?: string) => {
	if (!category || category === "all") {
		return skillsData;
	}
	return skillsData.filter((s) => s.category === category);
};

// Get advanced skills
export const getAdvancedSkills = () => {
	return skillsData.filter(
		(s) => s.level === "advanced" || s.level === "expert",
	);
};

// Calculate total years of experience
export const getTotalExperience = () => {
	const totalMonths = skillsData.reduce((total, skill) => {
		return total + skill.experience.years * 12 + skill.experience.months;
	}, 0);
	return {
		years: Math.floor(totalMonths / 12),
		months: totalMonths % 12,
	};
};
