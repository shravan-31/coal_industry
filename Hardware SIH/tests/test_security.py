import unittest
import os
import sqlite3
import json
from security import SecurityManager

class TestSecurityManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test security manager before each test."""
        self.db_file = "test_security.db"
        self.key_file = "test_encryption.key"
        self.security = SecurityManager(self.db_file)
    
    def tearDown(self):
        """Clean up test files after each test."""
        if os.path.exists(self.db_file):
            os.remove(self.db_file)
        if os.path.exists(self.key_file):
            os.remove(self.key_file)
    
    def test_init_security_database(self):
        """Test security database initialization."""
        # Check that database file was created
        self.assertTrue(os.path.exists(self.db_file))
        
        # Check that tables were created
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Check important tables
        tables_to_check = ['users', 'roles', 'sessions', 'audit_log', 'access_control']
        
        for table in tables_to_check:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            self.assertIsNotNone(cursor.fetchone(), f"Table {table} not found")
        
        conn.close()
    
    def test_hash_password(self):
        """Test password hashing."""
        password = "test_password123"
        hashed_password, salt = self.security.hash_password(password)
        
        self.assertIsInstance(hashed_password, str)
        self.assertIsInstance(salt, str)
        self.assertNotEqual(hashed_password, password)
        self.assertGreater(len(hashed_password), 0)
        self.assertGreater(len(salt), 0)
    
    def test_verify_password(self):
        """Test password verification."""
        password = "test_password123"
        hashed_password, salt = self.security.hash_password(password)
        
        # Verify correct password
        self.assertTrue(self.security.verify_password(password, hashed_password, salt))
        
        # Verify incorrect password
        self.assertFalse(self.security.verify_password("wrong_password", hashed_password, salt))
    
    def test_create_user(self):
        """Test creating a user."""
        result = self.security.create_user(
            user_id="test001",
            username="testuser",
            password="Test123!",
            role="reviewer",
            email="test@example.com"
        )
        
        self.assertTrue(result)
        
        # Try to create the same user again (should fail)
        result = self.security.create_user(
            user_id="test002",
            username="testuser",  # Same username
            password="Test123!",
            role="reviewer",
            email="test2@example.com"
        )
        
        self.assertFalse(result)
    
    def test_user_exists(self):
        """Test checking if user exists."""
        # User doesn't exist yet
        self.assertFalse(self.security.user_exists("nonexistent"))
        
        # Create user
        self.security.create_user("test001", "testuser", "Test123!", "reviewer")
        
        # User now exists
        self.assertTrue(self.security.user_exists("testuser"))
    
    def test_authenticate_user(self):
        """Test user authentication."""
        # Create a user
        self.security.create_user("test001", "testuser", "Test123!", "reviewer")
        
        # Authenticate with correct credentials
        user_info = self.security.authenticate_user("testuser", "Test123!")
        self.assertIsNotNone(user_info)
        self.assertEqual(user_info['username'], "testuser")
        self.assertEqual(user_info['role'], "reviewer")
        
        # Authenticate with incorrect password
        user_info = self.security.authenticate_user("testuser", "Wrong123!")
        self.assertIsNone(user_info)
        
        # Authenticate with non-existent user
        user_info = self.security.authenticate_user("nonexistent", "Test123!")
        self.assertIsNone(user_info)
    
    def test_create_and_validate_session(self):
        """Test session creation and validation."""
        # Create a user first
        self.security.create_user("test001", "testuser", "Test123!", "reviewer")
        user_info = self.security.authenticate_user("testuser", "Test123!")
        
        # Create session
        session_id = self.security.create_session(
            user_id="test001",
            ip_address="127.0.0.1",
            user_agent="Test Browser"
        )
        
        self.assertIsInstance(session_id, str)
        self.assertGreater(len(session_id), 0)
        
        # Validate session
        session_info = self.security.validate_session(session_id)
        self.assertIsNotNone(session_info)
        self.assertEqual(session_info['user_id'], "test001")
        self.assertEqual(session_info['username'], "testuser")
    
    def test_invalidate_session(self):
        """Test session invalidation."""
        # Create session
        session_id = self.security.create_session("test001")
        
        # Validate session (should work)
        session_info = self.security.validate_session(session_id)
        self.assertIsNotNone(session_info)
        
        # Invalidate session
        result = self.security.invalidate_session(session_id)
        self.assertTrue(result)
        
        # Validate session (should fail now)
        session_info = self.security.validate_session(session_id)
        self.assertIsNone(session_info)
    
    def test_has_permission(self):
        """Test permission checking."""
        # Admin should have all permissions
        self.assertTrue(self.security.has_permission("admin", "all"))
        self.assertTrue(self.security.has_permission("admin", "read_proposals"))
        
        # Create a reviewer user
        self.security.create_user("rev001", "reviewer", "Test123!", "reviewer")
        
        # Reviewer should have reviewer permissions
        self.assertTrue(self.security.has_permission("rev001", "read_proposals"))
        self.assertTrue(self.security.has_permission("rev001", "evaluate"))
        self.assertTrue(self.security.has_permission("rev001", "feedback"))
        
        # Reviewer should not have admin permissions
        self.assertFalse(self.security.has_permission("rev001", "all"))
    
    def test_log_audit_event(self):
        """Test audit event logging."""
        details = {"test": "data", "value": 123}
        log_id = self.security.log_audit_event(
            user_id="test001",
            action="test_action",
            resource="test_resource",
            details=details,
            ip_address="127.0.0.1",
            success=True
        )
        
        self.assertIsInstance(log_id, int)
        self.assertGreater(log_id, 0)
        
        # Verify log entry was created
        audit_logs = self.security.get_audit_log(user_id="test001")
        self.assertEqual(len(audit_logs), 1)
        log_entry = audit_logs[0]
        self.assertEqual(log_entry['action'], "test_action")
        self.assertEqual(log_entry['resource'], "test_resource")
        self.assertEqual(json.loads(log_entry['details']), details)
    
    def test_encrypt_decrypt_data(self):
        """Test data encryption and decryption."""
        original_data = "This is sensitive information that needs encryption"
        
        # Encrypt data
        encrypted_data = self.security.encrypt_data(original_data)
        self.assertIsInstance(encrypted_data, str)
        self.assertNotEqual(encrypted_data, original_data)
        
        # Decrypt data
        decrypted_data = self.security.decrypt_data(encrypted_data)
        self.assertEqual(decrypted_data, original_data)
    
    def test_grant_and_check_access(self):
        """Test access granting and checking."""
        # Grant access
        ac_id = self.security.grant_access(
            resource_type="proposal",
            resource_id="PROP001",
            user_id="test001",
            permission_level="review",
            granted_by="admin"
        )
        
        self.assertIsInstance(ac_id, int)
        self.assertGreater(ac_id, 0)
        
        # Check access
        has_access = self.security.check_resource_access(
            resource_type="proposal",
            resource_id="PROP001",
            user_id="test001",
            required_permission="review"
        )
        
        self.assertTrue(has_access)

if __name__ == '__main__':
    unittest.main()